import os
import csv
import torch
from settings import INSTANCE_FOLDER, EXPERIMENTS_FOLDER

def run_eval(solver, folder, H, max_steps):
    # 1. Configuración de hilos para pureza en el benchmarking
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # 2. Nombre y ruta del archivo de salida
    folder_name = os.path.basename(os.path.normpath(folder))
    # Verificamos si el solver tiene el atributo 'w'
    if hasattr(solver, 'w'):
        csv_filename = f"{solver.name}_w{solver.w}_{folder_name}.csv"
        w_info = f" | Parámetro w: {solver.w}"
    else:
        csv_filename = f"{solver.name}_{folder_name}.csv"
        w_info = ""
    
    # Combinamos la carpeta de experimentos con el nombre del archivo
    csv_filepath = EXPERIMENTS_FOLDER / csv_filename
    
    # Asegurarnos de que el directorio de experimentos exista
    EXPERIMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # 3. Preparación de instancias
    folder_path = os.path.join(INSTANCE_FOLDER, folder)
    instances = sorted(os.listdir(folder_path))  # sorted para determinismo
    total = len(instances)

    print(f"--- Iniciando benchmarking ---")
    print(f"Solver: {solver.name}{w_info}")
    print(f"Guardando resultados en: {csv_filepath}\n")

    # 4. Abrir archivo en modo escritura y procesar
    with open(csv_filepath, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Instance', 'Solved', 'Steps', 'Time'])

        for i, instance in enumerate(instances, start=1):
            # Resolver la instancia
            instance_path = os.path.join(folder, instance)
            solved, steps, t = solver.solve(instance_path, H, max_steps)

            # Guardar en CSV incrementalmente
            writer.writerow([instance, solved, steps, t])
            csv_file.flush()  # Fuerza la escritura a disco inmediatamente

            # Imprimir progreso simplificado por consola
            print(f"[{i:>2}/{total}] {instance:<15} | Solved: {str(solved):<5} | Steps: {steps:<4} | Time: {t:.4f}s")

    print("\nEvaluación finalizada.")