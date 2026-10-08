import os
import argparse
import shutil
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
    parser = argparse.ArgumentParser(description="Comprime y sube carpetas al servidor remoto.")
    parser.add_argument("carpetas", nargs="+", help="Rutas a las carpetas locales")
    args = parser.parse_args()

    print(f"Conectando a {USER}@{HOST}...")

    try:
        with Connection(host=HOST, user=USER) as c:
            # Asegurar que el directorio remoto base existe
            c.run(f"mkdir -p {DIRECTORIO_REMOTO}")

            for ruta_local in args.carpetas:
                if not os.path.isdir(ruta_local):
                    print(f"Omitiendo: '{ruta_local}' no es una carpeta válida.")
                    continue

                # 1. Preparar nombres
                nombre_base = os.path.basename(os.path.normpath(ruta_local))
                archivo_tar = f"{nombre_base}.tar.gz"
                ruta_remota_tar = f"{DIRECTORIO_REMOTO}/{archivo_tar}"

                print(f"\n--- Procesando: {nombre_base} ---")
                
                # 2. Comprimir localmente
                print(f"Comprimiendo carpeta en {archivo_tar}...")
                shutil.make_archive(nombre_base, 'gztar', root_dir=ruta_local)

                try:
                    # 3. Subir archivo
                    print(f"Subiendo a {HOST}...")
                    c.put(archivo_tar, remote=ruta_remota_tar)

                    # 4. Descomprimir en el servidor
                    # -C cambia al directorio antes de extraer
                    print(f"Descomprimiendo en el servidor...")
                    c.run(f"mkdir -p {DIRECTORIO_REMOTO}/{nombre_base}")
                    c.run(f"tar -xzf {ruta_remota_tar} -C {DIRECTORIO_REMOTO}/{nombre_base}")

                    # 5. Limpiar archivo temporal en el servidor
                    c.run(f"rm {ruta_remota_tar}")
                    print(f"¡Éxito! Carpeta disponible en: {DIRECTORIO_REMOTO}/{nombre_base}")

                finally:
                    # 6. Limpiar archivo temporal local (siempre se ejecuta)
                    if os.path.exists(archivo_tar):
                        os.remove(archivo_tar)

            print("\nProceso finalizado.")

    except Exception as e:
        print(f"\n[ERROR]: {e}")

if __name__ == "__main__":
    main()