"""
Mapeo de motivos de fallo de VEXA a mensajes claros para el usuario.

Cuando una reunion falla (bot no pudo unirse, bot salio por error, etc.),
se guarda un mensaje en meeting.error_message para que el usuario vea
el motivo al entrar a la reunion fallida.
"""
from typing import Optional


# Claves que VEXA o el bot-manager pueden enviar en "reason" (case-insensitive)
VEXA_REASON_TO_MESSAGE = {
    "not_invited": "Nadie invito al bot a la reunion. Asegurate de admitir al bot cuando entre a la llamada.",
    "no_participants": "Nadie se presento a la reunion en el tiempo esperado. El bot salio automaticamente.",
    "timeout": "El bot salio porque nadie se presento a la reunion en el tiempo configurado.",
    "empty_meeting": "La reunion quedo vacia. El bot salio al no haber participantes.",
    "connection_error": "Problemas de conexion. El bot no pudo mantener la conexion con la reunion.",
    "kicked": "El bot fue expulsado de la reunion.",
    "meeting_ended": "La reunion termino antes de que el bot pudiera grabar.",
    "rejected": "La reunion rechazo la entrada del bot (por ejemplo, reunion ya llena o restringida).",
    "unable_to_join": "El bot no pudo unirse a la reunion. Comprueba el enlace y el codigo de acceso.",
    "invalid_link": "El enlace de la reunion no es valido o ha expirado.",
    "api_error": "Error del servicio del bot. Intentalo de nuevo mas tarde.",
    "unknown": "El bot no pudo completar la reunion por un motivo no especificado.",
}


def vexa_failure_reason_to_message(reason: Optional[str]) -> str:
    """
    Convierte el motivo de fallo devuelto por VEXA (o bot-manager) a un mensaje
    claro en espanol para mostrar al usuario.

    Args:
        reason: Razon tal como viene en el webhook (payload.reason) o de la API.

    Returns:
        Mensaje en espanol para meeting.error_message.
    """
    if not reason or not str(reason).strip():
        return (
            "La reunion termino en estado fallido. "
            "Posibles causas: nadie invito al bot, nadie se presento a la reunion, "
            "el bot salio por tiempo, o hubo problemas de conexion."
        )
    key = str(reason).strip().lower().replace(" ", "_").replace("-", "_")
    # Buscar coincidencia exacta
    if key in VEXA_REASON_TO_MESSAGE:
        return VEXA_REASON_TO_MESSAGE[key]
    # Buscar si alguna clave esta contenida en reason
    for code, msg in VEXA_REASON_TO_MESSAGE.items():
        if code in key or code.replace("_", " ") in reason.lower():
            return msg
    # Si el reason ya parece un mensaje legible (mas de 10 chars, contiene espacios), usarlo
    if len(reason) > 10 and " " in reason:
        return reason
    return VEXA_REASON_TO_MESSAGE["unknown"]
