import os
import argparse
from dotenv import load_dotenv
from fabric import Connection
from pathlib import Path

# Cargar variables de entorno
load_dotenv()

USER = os.getenv("user")
HOST = os.getenv("host")

# --- CONFIGURACIÓN LOCAL ---

DESTINO_LOCAL = Path(__file__).resolve().parent
DIRECTORIO_REMOTO = f"/work/{USER}/CPMP"

# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Recupera archivos del servidor remoto vía SSH.")
    parser.add_argument("archivos_remotos", nargs="+", help="Rutas completas de los archivos en el servidor")
    args = parser.parse_args()

    print(f"Conectando a {USER}@{HOST}...")

    try:
        with Connection(host=HOST, user=USER) as c:
            for ruta_remota in args.archivos_remotos:
                ruta_remota = f"{DIRECTORIO_REMOTO}/{ruta_remota}"

                # Extraemos solo el nombre del archivo para la ruta local
                nombre_archivo = os.path.basename(ruta_remota)
                ruta_local_final = os.path.join(DESTINO_LOCAL, nombre_archivo)

                print(f"Descargando: {ruta_remota} -> {ruta_local_final}")
                
                try:
                    # El método .get() descarga del servidor al PC local
                    c.get(ruta_remota, local=ruta_local_final)
                except Exception as file_error:
                    print(f"Error al descargar {ruta_remota}: {file_error}")
            
            print(f"\nProceso finalizado. Archivos guardados en: {os.path.abspath(DESTINO_LOCAL)}")

    except Exception as e:
        print(f"\n[ERROR DE CONEXIÓN]: {e}")

if __name__ == "__main__":
    main()