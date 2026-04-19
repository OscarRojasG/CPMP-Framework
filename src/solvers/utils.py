def summary(results):
    solved = [r[0] for r in results]
    steps = [r[1] for r in results]

    solved_count = sum(solved)
    total_count = len(solved)

    print(f"Instancias resueltas: {solved_count}/{total_count}")

    steps_solved = [s for i, s in enumerate(steps) if solved[i]]
    steps_avg = sum(steps_solved) / len(steps_solved)

    print(f"Promedio de pasos: {steps_avg:.2f}")