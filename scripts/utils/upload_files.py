import os
import argparse
from dotenv import load_dotenv
from fabric import Connection

# Cargar variables de entorno
load_dotenv()

USER = os.getenv("user")
HOST = os.getenv("host")

# --- CONFIGURACIÓN ---

DIRECTORIO_REMOTO = f"/work/{USER}/CPMP"

# ---------------------

def main():
    parser = argparse.ArgumentParser(description="Sube archivos vía SSH usando Fabric.")
    parser.add_argument("archivos", nargs="+", help="Archivos locales a subir")
    args = parser.parse_args()

    print(f"Conectando a {USER}@{HOST}...")

    try:
        # La conexión se abre al entrar al bloque 'with' o al ejecutar una acción
        with Connection(host=HOST, user=USER) as c:
            for ruta in args.archivos:
                if os.path.exists(ruta):
                    print(f"Subiendo: {ruta}...")
                    c.put(ruta, remote=DIRECTORIO_REMOTO)
                else:
                    print(f"Archivo no encontrado: {ruta}")
            
            print("\nFinalizado.")

    except Exception as e:
        # Captura errores de conexión, autenticación o permisos
        print(f"\n[ERROR]: No se pudo completar la operación.")
        print(f"Detalle: {e}")

if __name__ == "__main__":
    main()