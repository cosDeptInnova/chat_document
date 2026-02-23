# device_manager.py
import os
from contextlib import contextmanager
from typing import List, Optional, Dict, Tuple
import threading
import time
import torch


class _RWLock:
    """
    RWLock simple:
    - múltiples readers simultáneos (inferencia)
    - un writer exclusivo (model.to / cambio de device)
    """
    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers <= 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


class _PoolLimiter:
    """
    Semáforos por (pool, gpu_index).

    CAMBIO CRÍTICO:
    - La capacidad se calcula sobre *presupuesto real disponible*,
      NO sobre VRAM total. Esto es imprescindible cuando las GPUs ya
      están ocupadas por otros servicios (LLMs grandes).

    Además:
    - Se hace un check de memoria libre "en caliente" en cada acquire
      para evitar que el semáforo permita pasar aunque haya bajado la VRAM.
    """
    def __init__(self, allowed_devices: List[int]) -> None:
        self.allowed_devices = list(allowed_devices)
        self._lock = threading.Lock()
        self._sems: Dict[str, Dict[int, threading.BoundedSemaphore]] = {}
        self._caps: Dict[str, Dict[int, int]] = {}
        self._budget_mb: Dict[int, int] = {}
        self._init_from_budget()

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)).strip())
        except Exception:
            return default

    def _pool_cost_mb(self, pool: str) -> int:
        # Costes conservadores por defecto (ajustables por env).
        defaults = {
            "embed":   1800,
            "rerank":  2500,
            "colbert": 3000,
            "splade":  2500,
            "ingest":  2500,
        }
        key = pool.upper()
        return self._env_int(f"RAG_POOL_{key}_MB", defaults.get(pool, 2000))

    def required_mb(self, pool: str) -> int:
        return max(256, int(self._pool_cost_mb(pool)))

    def _gpu_reserved_mb(self, idx: int) -> int:
        # Reserva por GPU (permite offsets distintos)
        # - RAG_GPU_RESERVED_MB_0=xxxx
        # - RAG_GPU_RESERVED_MB_1=xxxx
        # fallback: RAG_GPU_RESERVED_MB
        per = self._env_int(f"RAG_GPU_RESERVED_MB_{idx}", -1)
        if per >= 0:
            return per
        return self._env_int("RAG_GPU_RESERVED_MB", 0)

    def _gpu_budget_mb(self, idx: int) -> int:
        """
        Presupuesto usable para RAG en esa GPU.

        Prioridad:
        1) RAG_GPU_BUDGET_MB_<idx> (fijo)
        2) RAG_GPU_BUDGET_MB       (fijo, global)
        3) Memoria libre *actual* al inicializar el proceso (mem_get_info)
           menos reserva configurable.
        """
        per = self._env_int(f"RAG_GPU_BUDGET_MB_{idx}", -1)
        if per > 0:
            return per

        global_budget = self._env_int("RAG_GPU_BUDGET_MB", -1)
        if global_budget > 0:
            return global_budget

        # fallback: usar free actual al arranque
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_mb = int(free_bytes / (1024 ** 2))
        except Exception:
            free_mb = 0

        reserved = self._gpu_reserved_mb(idx)
        return max(0, free_mb - reserved)

    def _init_from_budget(self) -> None:
        baseline_mb = self._env_int("RAG_POOL_BASELINE_MB", 512)
        min_cap = max(0, self._env_int("RAG_POOL_MIN_CONCURRENCY", 1))  # permite 0
        max_cap = self._env_int("RAG_POOL_MAX_CONCURRENCY", 0)  # 0 = sin tope

        pools = ["embed", "rerank", "colbert", "splade", "ingest"]
        caps: Dict[str, Dict[int, int]] = {p: {} for p in pools}
        budgets: Dict[int, int] = {}

        for idx in self.allowed_devices:
            budget_mb = self._gpu_budget_mb(idx)
            budgets[idx] = budget_mb

            for pool in pools:
                cost = self.required_mb(pool)
                usable = max(0, budget_mb - baseline_mb)

                if budget_mb <= 0:
                    cap = 0
                else:
                    cap = int(usable // max(1, cost))
                    # si usable < cost, cap=0 (no forzamos 1)
                    cap = max(min_cap, cap)

                if max_cap and cap > max_cap:
                    cap = max_cap

                caps[pool][idx] = int(cap)

        # construir semáforos sólo donde cap>0
        sems: Dict[str, Dict[int, threading.BoundedSemaphore]] = {p: {} for p in pools}
        for pool in pools:
            for idx in self.allowed_devices:
                cap = int(caps[pool].get(idx, 0))
                if cap > 0:
                    sems[pool][idx] = threading.BoundedSemaphore(value=cap)

        with self._lock:
            self._budget_mb = budgets
            self._caps = caps
            self._sems = sems

    def capacity(self, pool: str, idx: int) -> int:
        with self._lock:
            return int(self._caps.get(pool, {}).get(idx, 0))

    def budget_mb(self, idx: int) -> int:
        with self._lock:
            return int(self._budget_mb.get(idx, 0))

    def acquire(self, pool: str, idx: int, timeout_s: float) -> bool:
        """
        Acquire conservador:
        - Si cap<=0 => False
        - Hot-check previo (si no hay VRAM libre suficiente, no esperamos).
        - Si mem_get_info falla: por defecto NO usamos esa GPU (configurable).
        - Hot-check posterior: si bajó la VRAM mientras esperábamos, soltamos.
        """
        allow_if_no_meminfo = self._env_int("RAG_POOL_ALLOW_IF_NO_MEMINFO", 0) == 1

        with self._lock:
            sem = self._sems.get(pool, {}).get(idx)
            cap = int(self._caps.get(pool, {}).get(idx, 0))

        if sem is None or cap <= 0:
            return False

        headroom = self._env_int("RAG_POOL_HEADROOM_MB", 256)
        needed = self.required_mb(pool) + max(0, headroom)

        # --- Hot check previo ---
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_mb = int(free_bytes / (1024 ** 2))
            if free_mb < needed:
                return False
        except Exception:
            if not allow_if_no_meminfo:
                return False

        ok = sem.acquire(timeout=timeout_s)
        if not ok:
            return False

        # --- Hot check posterior ---
        try:
            free_bytes2, _ = torch.cuda.mem_get_info(idx)
            free_mb2 = int(free_bytes2 / (1024 ** 2))
            if free_mb2 < needed:
                try:
                    sem.release()
                except Exception:
                    pass
                return False
        except Exception:
            if not allow_if_no_meminfo:
                try:
                    sem.release()
                except Exception:
                    pass
                return False

        return True

    def release(self, pool: str, idx: int) -> None:
        with self._lock:
            sem = self._sems.get(pool, {}).get(idx)
        if sem is None:
            return
        try:
            sem.release()
        except Exception:
            pass


class GPUDeviceManager:
    """
    Gestor centralizado de dispositivos para todo el RAG.

    Cambios clave para multiusuario real sin llenar VRAM:
    - PoolLimiter basado en *budget libre real* (no VRAM total)
    - Pinning por modelo (evita thrashing de model.to entre GPUs)
    - Check de VRAM libre en caliente antes de permitir entrada por pool
    """
    def __init__(self) -> None:
        self._model_locks: Dict[int, _RWLock] = {}
        self._model_locks_guard = threading.Lock()

        # “Home device” por modelo para evitar migraciones (thrash)
        self._model_home: Dict[int, torch.device] = {}
        self._model_home_guard = threading.Lock()

        self._init_devices()
        self._pool_limiter = _PoolLimiter(self.allowed_devices)

        # Por defecto: no migrar modelos entre GPUs salvo necesidad (OOM).
        # Puedes forzarlo con RAG_ALLOW_MODEL_MIGRATION=1
        self._allow_migration = os.getenv("RAG_ALLOW_MODEL_MIGRATION", "0").strip() in ("1", "true", "yes", "on")

        # Headroom mínimo para admitir una tarea en GPU una vez adquirido el semáforo
        self._pool_headroom_mb = self._env_int("RAG_POOL_HEADROOM_MB", 256)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)).strip())
        except Exception:
            return default

    def _get_model_lock(self, model: torch.nn.Module) -> _RWLock:
        mid = id(model)
        with self._model_locks_guard:
            lk = self._model_locks.get(mid)
            if lk is None:
                lk = _RWLock()
                self._model_locks[mid] = lk
            return lk

    def _init_devices(self) -> None:
        env = os.getenv("RAG_DEVICES", "0,1").strip()
        if torch.cuda.is_available():
            total = torch.cuda.device_count()
            if env:
                allowed: List[int] = []
                for s in env.split(","):
                    s = s.strip()
                    if not s:
                        continue
                    try:
                        i = int(s)
                    except ValueError:
                        continue
                    if 0 <= i < total:
                        allowed.append(i)
                self.allowed_devices = allowed or list(range(total))
            else:
                self.allowed_devices = list(range(total))
        else:
            self.allowed_devices = []

    def list_devices(self) -> List[int]:
        return list(self.allowed_devices)

    def best_device(self, min_free_mb: int = 2048) -> torch.device:
        if not self.allowed_devices:
            return torch.device("cpu")

        best_idx = None
        best_free_mb = 0.0

        for idx in self.allowed_devices:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except Exception:
                continue
            free_mb = free_bytes / (1024 ** 2)
            if free_mb >= min_free_mb and free_mb > best_free_mb:
                best_free_mb = free_mb
                best_idx = idx

        if best_idx is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{best_idx}")

    def _get_model_current_device(self, model: torch.nn.Module) -> Optional[torch.device]:
        try:
            p = next(model.parameters())
            return p.device
        except Exception:
            return None

    def _get_or_set_home_device(
        self,
        model: torch.nn.Module,
        prefer_device: Optional[torch.device],
        min_free_mb: int,
    ) -> torch.device:
        """
        Decide el “home device” para un modelo (por proceso) para evitar migraciones.

        - Si ya está en CUDA -> se considera home (siempre que esté en allowed)
        - Si no, usa prefer_device si CUDA y válida
        - Si no, elige best_device
        - Si todo falla -> CPU
        """
        mid = id(model)
        with self._model_home_guard:
            home = self._model_home.get(mid)
            if home is not None:
                return home

            cur = self._get_model_current_device(model)
            if cur is not None and cur.type == "cuda":
                # respeta el device actual si está permitido
                if cur.index in self.allowed_devices:
                    self._model_home[mid] = cur
                    return cur

            if prefer_device is not None and prefer_device.type == "cuda":
                if prefer_device.index in self.allowed_devices:
                    self._model_home[mid] = prefer_device
                    return prefer_device

            best = self.best_device(min_free_mb=min_free_mb)
            self._model_home[mid] = best
            return best

    def _iter_candidate_devices(
        self,
        home: torch.device,
        prefer_device: Optional[torch.device],
        fallback_to_cpu: bool,
    ) -> List[torch.device]:
        """
        Orden de candidatos:
        1) home
        2) prefer (si distinto)
        3) otras GPUs permitidas (ordenadas por memoria libre desc)
        4) CPU (si fallback_to_cpu)
        """
        out: List[torch.device] = []

        def _add(d: torch.device):
            if d is None:
                return
            if d.type == "cuda" and (d.index not in self.allowed_devices):
                return
            if all((x.type != d.type) or (x.type == "cuda" and x.index != d.index) for x in out):
                out.append(d)

        _add(home)
        if prefer_device is not None:
            _add(prefer_device)

        # Otras GPUs por free desc (best-effort)
        if self.allowed_devices:
            free_list: List[Tuple[float, int]] = []
            for idx in self.allowed_devices:
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(idx)
                    free_mb = float(free_bytes / (1024 ** 2))
                except Exception:
                    free_mb = 0.0
                free_list.append((free_mb, idx))
            free_list.sort(reverse=True, key=lambda x: x[0])

            for _, idx in free_list:
                _add(torch.device(f"cuda:{idx}"))

        if fallback_to_cpu:
            _add(torch.device("cpu"))

        return out

    @contextmanager
    def use_device_with_fallback(
        self,
        model: torch.nn.Module,
        prefer_device: Optional[torch.device] = None,
        min_free_mb: int = 1024,
        fallback_to_cpu: bool = True,
        *,
        pool: Optional[str] = None,
    ):
        """
        Context manager para ejecutar inferencia con fallback:
        - home_device (pin) -> prefer -> otras GPUs -> CPU
        + RWLock por modelo
        + semáforo dinámico por pool y GPU
        + chequeo de VRAM libre en caliente tras adquirir semáforo
        """
        model_lock = self._get_model_lock(model)

        home = self._get_or_set_home_device(model, prefer_device=prefer_device, min_free_mb=min_free_mb)
        candidate_devices = self._iter_candidate_devices(home, prefer_device, fallback_to_cpu)

        acquire_timeout = float(os.getenv("RAG_POOL_ACQUIRE_TIMEOUT_SEC", "2.0"))
        last_error: Optional[BaseException] = None

        for dev in candidate_devices:
            # 1) VRAM free check (best-effort)
            if dev.type == "cuda":
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(dev.index)
                    if free_bytes < min_free_mb * 1024 ** 2:
                        continue
                except Exception:
                    pass

            # 2) pool semaphore (si aplica) -> si no hay cupo, probamos otra
            sem_acquired = False
            if pool and dev.type == "cuda":
                ok = self._pool_limiter.acquire(pool, int(dev.index), timeout_s=acquire_timeout)
                if not ok:
                    continue
                sem_acquired = True

                # 2.1) Check “en caliente” de memoria libre suficiente para este pool
                #      (evita que el semáforo pase cuando la GPU ya está justa)
                try:
                    free_bytes2, _ = torch.cuda.mem_get_info(dev.index)
                    free_mb2 = int(free_bytes2 / (1024 ** 2))
                except Exception:
                    free_mb2 = 0

                needed = self._pool_limiter.required_mb(pool) + self._pool_headroom_mb
                if free_mb2 > 0 and free_mb2 < needed:
                    # no hay margen: soltamos y probamos otra GPU/CPU
                    self._pool_limiter.release(pool, int(dev.index))
                    sem_acquired = False
                    continue

            read_acquired = False
            try:
                # 3) mover modelo sólo si hace falta (write-lock)
                cur = self._get_model_current_device(model)

                need_move = (
                    (cur is None) or
                    (cur.type != dev.type) or
                    (cur.type == "cuda" and dev.type == "cuda" and cur.index != dev.index)
                )

                # Evitar migración entre GPUs si no está permitido (reduce thrash)
                if (not self._allow_migration) and cur is not None and cur.type == "cuda" and dev.type == "cuda":
                    if cur.index != dev.index:
                        # Si el modelo ya está en una GPU, no lo movemos a otra:
                        # probamos siguiente candidato.
                        continue

                if need_move:
                    model_lock.acquire_write()
                    try:
                        cur2 = self._get_model_current_device(model)
                        need_move2 = (
                            (cur2 is None) or
                            (cur2.type != dev.type) or
                            (cur2.type == "cuda" and dev.type == "cuda" and cur2.index != dev.index)
                        )
                        if need_move2:
                            model.to(dev)
                    finally:
                        model_lock.release_write()

                # 4) inferencia concurrente (read-lock)
                model_lock.acquire_read()
                read_acquired = True
                try:
                    yield dev
                    return
                except torch.cuda.OutOfMemoryError as e:
                    last_error = e
                    if dev.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue

            finally:
                if read_acquired:
                    try:
                        model_lock.release_read()
                    except Exception:
                        pass
                if sem_acquired and pool and dev.type == "cuda":
                    self._pool_limiter.release(pool, int(dev.index))

        if last_error is not None:
            raise last_error
        raise RuntimeError("No hay dispositivos disponibles para inferencia.")


gpu_manager = GPUDeviceManager()
