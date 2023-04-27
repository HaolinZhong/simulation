import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EGArm import EGArm
from TSArm import TSArm
from UCB1Arm import UCBArm
from tqdm import trange


def experiment(
        method,
        f_ac,
        nsimu=100000,
        nsize=80,
        p_truth=[0.3, 0.4, 0.5, 0.6],
        rho=0,
        eps=1
):
    p_truth.sort()

    n_best_choice = []
    n_pos_outcome = []
    best_identified = []

    for x in trange(nsimu):

        if method == "EG":
            arms = [EGArm(p) for p in p_truth]
        elif method == "UCB":
            arms = [UCBArm(p) for p in p_truth]
            UCBArm.global_N = 0
        elif method == "TS":
            arms = [TSArm(p) for p in p_truth]
        else:
            print('Method should be: EG, UCB or TS')
            return None

        cur_best_choice = 0
        cur_pos_outcome = 0

        for i in range(nsize):
            acc = f_ac(i, nsize, rho)
            opt_j = np.argmax([acc * arm.get_p_estimate() + (1 - acc) * arm.sample() for arm in arms])
            if i == 0:
                opt_j = np.random.randint(len(arms))

            if method == "EG" and np.random.random() < eps * (((nsize - i - 1) / nsize) ** rho):
                opt_j = np.random.randint(len(arms))

            if method == "UCB" and i < len(arms):
                opt_j = i

            if opt_j == len(arms) - 1:
                cur_best_choice += 1

            arm_chosen = arms[opt_j]
            res = arm_chosen.get_outcome()

            cur_pos_outcome += res
            arm_chosen.update(res)

        n_best_choice.append(cur_best_choice)
        n_pos_outcome.append(cur_pos_outcome)
        estimated_opt_arm = np.argmax([arm.get_p_estimate() for arm in arms])
        best_identified.append(estimated_opt_arm == len(arms) - 1)

    sim_res = {}
    sim_res['method'] = method
    sim_res['sample size'] = nsize
    sim_res['rho'] = rho
    sim_res['eps'] = eps
    sim_res['func'] = f_ac.__name__
    sim_res['p* mean'] = round(np.mean(n_best_choice) / nsize, 3)
    sim_res['p* std'] = round(np.std(np.divide(n_best_choice, nsize)), 3)
    sim_res['NS mean'] = round(np.mean(n_pos_outcome), 1)
    sim_res['NS std'] = round(np.std(n_pos_outcome), 1)
    sim_res['Popt mean'] = round(np.mean(best_identified), 3)
    sim_res['Popt std'] = round(np.std(best_identified), 3)
    sim_res['truth'] = ", ".join([str(p) for p in p_truth])

    return sim_res

def dicts_to_table(dicts):
    # Get the list of keys from the first dictionary
    keys = list(dicts[0].keys())

    # Create an empty list to hold the rows
    rows = []

    # Iterate over the dictionaries and add their values as rows
    for d in dicts:
        row = []
        for key in keys:
            row.append(d[key])
        rows.append(row)

    # Create the table using the keys as column headers and the rows as data
    table = []
    table.append(keys)
    table.extend(rows)

    # Return the table
    return table




if __name__ == '__main__':
    def linear(i, nsize, rho):
        return (i / nsize) ** rho

    def cubic(i, nsize, rho):
        return ((i / nsize - 1) ** 3 + 1) ** rho

    def naive(i, nsize, rho):
        return 0

    #1. G-N
    def G_N_exp():
        results = []
        for m in ['TS', 'UCB', 'EG']:
            for nsize in [40, 80, 120]:
                if m == 'TS':
                    for f in [linear, cubic, naive]:
                        res = experiment(m, f, rho=1, nsize=nsize)
                        results.append(res)
                elif m == 'EG':
                    for eps in [1, 0]:
                        res = experiment(m, f, rho=1, nsize=nsize, eps=eps)
                        results.append(res)
                else:
                    res = experiment(m, naive, nsize=nsize)
                    results.append(res)

        # Convert the dictionaries to a table
        table = dicts_to_table(results)

        # Create a DataFrame object from the table
        df = pd.DataFrame(table[1:], columns=table[0])
        df.to_csv('G-N.csv', index=False)


    # 2. Rho-G-N
    def Rho_G_N():
        results = []
        for m in ['TS', 'EG']:
            for nsize in [40, 80, 120]:
                for rho in [0.5, 1, 2, 3, 4, 5]:
                    if m == 'TS':
                        for f in [linear, cubic]:
                            res = experiment(m, f, rho=rho, nsize=nsize)
                            results.append(res)
                    elif m == 'EG':
                        res = experiment(m, naive, rho=rho, nsize=nsize)
                        results.append(res)

        # Convert the dictionaries to a table
        table = dicts_to_table(results)

        # Create a DataFrame object from the table
        df = pd.DataFrame(table[1:], columns=table[0])
        df.to_csv('Rho-G-N.csv', index=False)

    # 3. vary truth
    def varying_truth():
        results = []
        for truth in [[0.3, 0.7], [0.4, 0.6], [0.45, 0.55]]:
            for m in ['TS', 'EG']:
                if m == 'TS':
                    for rho in [1, 2, 3]:
                        res = experiment(m, cubic, rho=rho, nsize=80, p_truth=truth)
                        results.append(res)
                    for rho in [0.5, 1, 2]:
                        res = experiment(m, linear, rho=rho, nsize=80, p_truth=truth)
                        results.append(res)
                elif m == 'EG':
                    for rho in [3, 4, 5]:
                        res = experiment(m, naive, rho=rho, nsize=80, p_truth=truth)
                        results.append(res)

        # Convert the dictionaries to a table
        table = dicts_to_table(results)

        # Create a DataFrame object from the table
        df = pd.DataFrame(table[1:], columns=table[0])
        df.to_csv('VT.csv', index=False)
