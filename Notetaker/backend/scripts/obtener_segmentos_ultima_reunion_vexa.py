"""
Script para obtener los segmentos de la ultima reunion desde VEXA.

Uso:
    python scripts/obtener_segmentos_ultima_reunion_vexa.py

Opcionalmente puedes especificar la plataforma:
    python scripts/obtener_segmentos_ultima_reunion_vexa.py --platform teams
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Cargar .env
env_file = backend_dir / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v

from app.services.vexa_service import VexaService, VexaServiceError
import argparse

def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parsea una fecha ISO desde VEXA."""
    if not dt_str:
        return None
    try:
        # Manejar formato con Z
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description='Obtener segmentos de la ultima reunion desde VEXA')
    parser.add_argument('--platform', type=str, default=None, help='Plataforma a filtrar (teams, google_meet)')
    parser.add_argument('--output', type=str, default=None, help='Archivo JSON donde guardar los segmentos')
    args = parser.parse_args()
    
    try:
        vexa = VexaService()
        print("Obteniendo lista de reuniones desde VEXA...")
        
        # Listar todas las reuniones
        meetings = vexa.list_meetings(platform=args.platform)
        
        if not meetings:
            print("No se encontraron reuniones en VEXA.")
            return
        
        print(f"Encontradas {len(meetings)} reuniones.")
        
        # Encontrar la mas reciente (por created_at o end_time)
        latest_meeting = None
        latest_time = None
        
        for meeting in meetings:
            # Intentar usar end_time primero, luego created_at
            meeting_time = None
            
            # Buscar end_time en diferentes ubicaciones posibles
            end_time_str = meeting.get("end_time") or meeting.get("actual_end_time")
            if end_time_str:
                meeting_time = parse_datetime(end_time_str)
            
            # Si no hay end_time, usar created_at
            if not meeting_time:
                created_at_str = meeting.get("created_at") or meeting.get("start_time")
                if created_at_str:
                    meeting_time = parse_datetime(created_at_str)
            
            if meeting_time:
                if latest_time is None or meeting_time > latest_time:
                    latest_time = meeting_time
                    latest_meeting = meeting
        
        if not latest_meeting:
            print("No se pudo determinar la reunion mas reciente.")
            return
        
        print(f"\nReunion mas reciente encontrada:")
        print(f"  ID interno: {latest_meeting.get('id')}")
        print(f"  Plataforma: {latest_meeting.get('platform')}")
        print(f"  Native Meeting ID: {latest_meeting.get('platform_specific_id')}")
        print(f"  Estado: {latest_meeting.get('status')}")
        print(f"  Fecha creacion: {latest_meeting.get('created_at')}")
        print(f"  Fecha fin: {latest_meeting.get('end_time') or latest_meeting.get('actual_end_time')}")
        
        native_id = latest_meeting.get("platform_specific_id")
        platform = latest_meeting.get("platform", "teams")
        
        if not native_id:
            print("Error: La reunion no tiene platform_specific_id (native_meeting_id).")
            return
        
        print(f"\nObteniendo transcripcion para native_id={native_id}, platform={platform}...")
        
        # Obtener transcripcion
        try:
            transcript_data = vexa.get_transcript(
                native_meeting_id=native_id,
                platform=platform,
            )
            
            segments = transcript_data.get("segments", [])
            
            if not segments:
                print("La reunion no tiene segmentos aun.")
                print(f"Datos completos de la reunion: {json.dumps(latest_meeting, indent=2, default=str)}")
                return
            
            print(f"\nEncontrados {len(segments)} segmentos.")
            
            # Mostrar resumen
            print("\nResumen de segmentos:")
            print(f"  Total segmentos: {len(segments)}")
            
            if segments:
                first_segment = segments[0]
                last_segment = segments[-1]
                print(f"  Primer segmento: {first_segment.get('absolute_start_time')} - {first_segment.get('text', '')[:50]}...")
                print(f"  Ultimo segmento: {last_segment.get('absolute_end_time')} - {last_segment.get('text', '')[:50]}...")
                
                # Contar speakers
                speakers = set()
                for seg in segments:
                    speaker = seg.get('speaker')
                    if speaker:
                        speakers.add(speaker)
                print(f"  Speakers unicos: {len(speakers)} - {', '.join(sorted(speakers))}")
            
            # Guardar en archivo si se especifica
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'meeting': latest_meeting,
                        'segments': segments,
                        'total_segments': len(segments),
                    }, f, indent=2, default=str, ensure_ascii=False)
                print(f"\nSegmentos guardados en: {output_path}")
            else:
                # Mostrar primeros y ultimos segmentos
                print("\nPrimeros 5 segmentos:")
                for i, seg in enumerate(segments[:5], 1):
                    print(f"  {i}. [{seg.get('absolute_start_time')}] {seg.get('speaker', 'Unknown')}: {seg.get('text', '')[:80]}")
                
                if len(segments) > 5:
                    print(f"\n... ({len(segments) - 5} segmentos mas) ...")
                    print("\nUltimos 5 segmentos:")
                    for i, seg in enumerate(segments[-5:], len(segments) - 4):
                        print(f"  {i}. [{seg.get('absolute_start_time')}] {seg.get('speaker', 'Unknown')}: {seg.get('text', '')[:80]}")
                
                print("\nPara guardar todos los segmentos en un archivo JSON, usa:")
                print(f"  python {sys.argv[0]} --output segmentos.json")
        
        except VexaServiceError as e:
            print(f"Error obteniendo transcripcion: {e}")
            return
    
    except VexaServiceError as e:
        print(f"Error de VEXA: {e}")
        return
    except Exception as e:
        print(f"Error inesperado: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()
