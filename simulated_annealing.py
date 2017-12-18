"""Simulated annealing algorithm on TSP"""
import numpy as np

from dataset import n, matrix


def close_perms(perm):
    for i in range(n):
        for j in range(i):
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            yield new_perm


def road_length(perm):
    length = matrix[perm[-1], perm[0]]
    for i in range(n - 1):
        length += matrix[perm[i], perm[i+1]]
    return length


def anneal(best_perm, best_score, max_evaluations):
    α = 0.998
    temp = 100
    num_evaluations = 1

    while num_evaluations < max_evaluations:
        # examine moves around our current position
        move_made = False
        for perm in close_perms(best_perm):
            if num_evaluations >= max_evaluations:
                break

            # see if this move is better than the current
            next_score = road_length(perm)
            num_evaluations += 1
            if np.exp(-(next_score-best_score)/temp) > np.random.rand():
                best_perm = perm
                best_score = next_score
                move_made = True
                break  # depth first search
            temp *= α
        if not move_made:
            break  # we couldn't find a better move
            # (must be at a local maximum)

    return num_evaluations, best_score, best_perm


if __name__ == '__main__':
    random_perm = np.random.permutation(n)
    score = road_length(random_perm)
    evals, best, perm = anneal(random_perm, score, 1000)

    print(f"Ran {evals} evaluations...")
    print("Shortest route found is:")
    print(perm)
    print("With length:")
    print(best)
