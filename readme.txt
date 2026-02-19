{
  "document_type": "codex_agent_runbook",
  "schema_version": "1.0.0",
  "language": "es",
  "platform": {
    "name": "chat_document multiagente",
    "core_objective": "Evolucionar y depurar una plataforma multiagente con orquestación MCP y tools reales, manteniendo UX con latencias bajas bajo concurrencia multiusuario.",
    "non_breaking_principle": "No romper contratos existentes de API, tokens, ni flujos planner->tools->analyst->writer->reviewer."
  },
  "system_topology": {
    "services": [
      {
        "name": "cosmos_mcp",
        "role": "bus de tools MCP + registry + auth",
        "critical_paths": [
          "cosmos_registry_list_tools",
          "call_tool",
          "auth headers S2S + X-User-Authorization"
        ]
      },
      {
        "name": "legal_search",
        "role": "orquestador multiagente legal con uso de tools remotas",
        "critical_paths": [
          "planner",
          "tool discovery",
          "search/fetch execution",
          "analyst loop",
          "writer/reviewer"
        ]
      },
      {
        "name": "chat_document/modeloNegocio",
        "role": "consumidores y patrones de referencia para tokens y MCP",
        "critical_paths": [
          "normalización de URL MCP",
          "propagación de token de servicio",
          "integración de sesiones de usuario"
        ]
      }
    ]
  },
  "latency_and_ux_targets": {
    "user_visible_slo": {
      "p50_end_to_end_seconds": 5,
      "p95_end_to_end_seconds": 12,
      "p99_end_to_end_seconds": 20
    },
    "tool_level_budgets": {
      "list_tools_p95_ms": 800,
      "search_tool_p95_seconds": 4,
      "fetch_tool_p95_seconds": 6,
      "single_iteration_soft_cap_seconds": 45
    },
    "degradation_policy": {
      "on_timeout": "reducir top_k, reducir max_iters, mantener respuesta útil con fuentes parciales",
      "on_tool_failure": "fallback controlado + eventos de diagnóstico + no bloquear todo el flujo salvo tools críticas ausentes"
    }
  },
  "concurrency_strategy": {
    "multiuser_model": {
      "isolation": "cada request conserva user_auth_token; S2S permanece en Authorization",
      "resource_control": [
        "semaforos por tipo de tool",
        "timeouts por llamada y por iteración",
        "backoff exponencial con jitter"
      ]
    },
    "recommended_defaults": {
      "LEGALSEARCH_MAX_CONCURRENT_FETCH": 6,
      "LEGALSEARCH_TOOL_TIMEOUT_SEC": 60,
      "LEGALSEARCH_ITER_TIMEOUT_SEC": 300,
      "LEGALSEARCH_TOOL_RETRIES": 2,
      "LEGALSEARCH_TOOL_BACKOFF_BASE": 0.5,
      "LEGALSEARCH_MAX_SOURCES_PER_ITER": 8,
      "LEGALSEARCH_MAX_SOURCES_TOTAL": 12
    },
    "autoscaling_notes": {
      "horizontal": "escalar réplicas de legal_search y cosmos_mcp según CPU, latencia p95 y cola de requests",
      "vertical": "aumentar CPU para parsing/fetch y memoria para bursts de sesiones",
      "load_shedding": "rechazo temprano con mensaje de alta demanda cuando se supera umbral crítico"
    }
  },
  "machine_debug_playbook": [
    {
      "id": "dbg-001",
      "goal": "Validar discoverability real de tools por crew/tag",
      "steps": [
        "Listar tools vía cosmos_registry_list_tools con crew=web_search_crew",
        "Comparar con list_tools estándar y detectar diferencias",
        "Validar metadata for_crews/tags en minúsculas y mayúsculas"
      ],
      "expected": "search/fetch disponibles y consistentes"
    },
    {
      "id": "dbg-002",
      "goal": "Confirmar propagación de auth dual (S2S + user)",
      "steps": [
        "Verificar header Authorization en request al MCP",
        "Verificar header X-User-Authorization en list_tools e invoke_tool",
        "Comprobar que el usuario no ve/ejecuta tools fuera de su alcance"
      ],
      "expected": "aislamiento multiusuario correcto"
    },
    {
      "id": "dbg-003",
      "goal": "Medir impacto de concurrencia en latencia UX",
      "steps": [
        "Ejecutar prueba de carga gradual (N usuarios concurrentes)",
        "Registrar p50/p95/p99 end-to-end y por tool",
        "Ajustar semáforos y top_k/max_iters para recuperar SLA"
      ],
      "expected": "sin degradación abrupta por picos"
    }
  ],
  "safe_change_protocol": {
    "before_change": [
      "identificar contrato de entrada/salida del módulo afectado",
      "crear checklist de reversibilidad",
      "preparar feature flag si el cambio impacta ruta crítica"
    ],
    "during_change": [
      "mantener backward compatibility de parámetros",
      "agregar logs estructurados con correlation_id/session_id",
      "no eliminar fallbacks sin reemplazo equivalente"
    ],
    "after_change": [
      "compilar módulos modificados",
      "probar flujo feliz y flujo degradado",
      "documentar tuning recomendado para producción"
    ]
  },
  "observability_contract": {
    "minimum_logs": [
      "tool_name",
      "attempt",
      "duration_ms",
      "status",
      "crew_name",
      "search_session_id",
      "conversation_id"
    ],
    "minimum_metrics": [
      "requests_total",
      "errors_total",
      "latency_seconds_bucket",
      "tool_calls_total",
      "tool_duration_seconds",
      "redis_latency_seconds",
      "db_latency_seconds"
    ],
    "alerts": [
      "p95_end_to_end > 12s durante 10m",
      "error_rate > 3% durante 5m",
      "timeout_ratio_tools > 5% durante 10m"
    ]
  },
  "agent_execution_policy": {
    "for_codex_agents": [
      "priorizar cambios pequeños, medibles y reversibles",
      "si falta una tool crítica, responder con diagnóstico estructurado",
      "usar discovery por metadata (crew/tags) antes de hardcodear nombres",
      "preservar coherencia entre list_tools e invoke_tool"
    ],
    "anti_patterns": [
      "aumentar timeouts de forma ilimitada",
      "ocultar errores de auth con retries infinitos",
      "mezclar token de usuario en Authorization S2S"
    ]
  },
  "next_iterations": [
    {
      "priority": "high",
      "item": "circuit breaker por tool",
      "value": "evita cascadas de timeout y protege p95"
    },
    {
      "priority": "high",
      "item": "colas por tenant/usuario",
      "value": "fairness bajo picos y mejor control de latencia"
    },
    {
      "priority": "medium",
      "item": "cache de discovery por crew+scope",
      "value": "menos RTT al MCP y arranque más rápido"
    },
    {
      "priority": "medium",
      "item": "benchmark reproducible con perfiles de carga",
      "value": "tuning basado en datos, no intuición"
    }
  ]
}
