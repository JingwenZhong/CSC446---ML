#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4/14/2021

@author: Jingwen
"""
import numpy as np

if not __file__.endswith('_em_gaussian.py'):
    print(
        'ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = '/u/cs246/data/em/'  # TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)


###
def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9 * len(data))
    train_xs = np.asarray(data[:dev_cutoff], dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:], dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs


###
def init_model(args):
    if args.cluster_num:
        # args.cluster_num is K
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num, 2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num, 2, 2))
        else:
            sigmas = np.zeros((2, 2))

        # TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        for i in range(args.cluster_num):
            lambdas[i] = np.random.uniform(0, 1, 1)  # use uniform distribution to control lambdas in range (0,1)
            mus[i] = np.random.normal(0, 1, 2)  # use normal distribution with mean=1, sigma = 2

        if not args.tied:
            for i in range(args.cluster_num):
                sigmas[i] = np.eye(2)
        else:
            sigmas = np.eye(2)  # array([[1., 0.],[0., 1.]])

    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file, 'r') as f:
            for line in f:
                # each line is a cluster, and looks like this:
                # lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float, line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)  # note here we have args.cluster_num

    # TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    # NOTE: if args.tied was provided, sigmas will have a different shape
    model = (lambdas, mus, sigmas)

    return model


###
def extract_parameters(model):
    # TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    return lambdas, mus, sigmas



###
def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    # NOTE: you can use multivariate_normal like this:
    # probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    # which is also the conditional distribution of x given a particular value for z
    # see text book page 431

    # TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    lambdas, mus, sigmas = extract_parameters(model)  # initial parameter

    # because the first log likelihood is very low(lower than -7), it is hard to observe later results so I don't
    # plot it. likelihood_train = [average_log_likelihood(model, train_xs, args)] if not args.nodev: likelihood_dev =
    # [average_log_likelihood(model, dev_xs, args)]

    likelihood_train =[]
    likelihood_dev =[]


    for iteration in range(args.iterations):
        if not args.tied:
            # E step
            Z_final = []
            for n in range(len(train_xs)):
                Z_top = []
                for i in range(args.cluster_num):
                    P_XZ = lambdas[i] * multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(train_xs[n])
                    Z_top.append(P_XZ)
                Z_top = np.array(Z_top)  # numerator which is P(x,z)
                Z_bottom = sum(Z_top)  # denominator which is P(x) =∑_k λ_k * N(X^(n)|μ_k,Σ_k)
                Z = Z_top / Z_bottom  # Z
                Z_final.append(Z)

            Z_final = np.array(Z_final)

            # M step
            for i in range(args.cluster_num):
                lambdas[i] = sum(Z_final[:, i]) / len(train_xs)  # update lambda
                mus[i] = np.dot(Z_final[:, i], train_xs) / sum(Z_final[:, i])  # update mu
                sigmas[i] = np.dot(Z_final[:, i] * np.transpose(train_xs - mus[i]), (train_xs - mus[i])) / sum(
                    Z_final[:, i])  # update sigma

        else:  # args.tied
            # E step
            Z_final = []
            for n in range(len(train_xs)):
                Z_top = []
                for i in range(args.cluster_num):
                    P_XZ = lambdas[i] * multivariate_normal(mean=mus[i], cov=sigmas).pdf(train_xs[n])
                    Z_top.append(P_XZ)
                Z_top = np.array(Z_top)  # numerator
                Z_bottom = sum(Z_top)  # denominator
                Z = Z_top / Z_bottom  # Z
                Z_final.append(Z)

            Z_final = np.array(Z_final)
            # M step
            sigmas = np.zeros((2, 2))
            for i in range(args.cluster_num):
                lambdas[i] = sum(Z_final[:, i]) / len(train_xs)  # update lambda
                mus[i] = np.dot(Z_final[:, i], train_xs) / sum(Z_final[:, i])  # update mu
                sigmas += np.dot(Z_final[:, i] * np.transpose(train_xs - mus[i]), (train_xs - mus[i]))
            sigmas = sigmas / len(train_xs)

        model = (lambdas, mus, sigmas)  # update model

        if not args.nodev:  # if dev set
            # append likelihood for train_set
            likelihood_train.append(average_log_likelihood(model, train_xs, args))

            # append likelihood for train_set
            likelihood_dev.append(average_log_likelihood(model, dev_xs, args))

    if not args.nodev:
        # plot:
        import matplotlib.pyplot as plt
        Total_iteration = np.arange(0, args.iterations, 1)
        plt.figure(figsize=(15, 7))
        plt.plot(Total_iteration, likelihood_train, marker='o', linestyle='-', label="training set")
        plt.plot(Total_iteration, likelihood_dev, marker='o', linestyle='-', label="dev set")
        plt.title('Iteration = ' + str(args.iterations) + ', K = ' + str(args.cluster_num))
        plt.xlabel("Iterations")
        plt.ylabel("Log likelihood")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig("Zhong_em_gaussian_ll.jpg")
        plt.show()

    return model


###
def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    # TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)

    lambdas, mus, sigmas = extract_parameters(model)
    log_px = []
    for n in range(len(data)):
        Z_top = []
        for i in range(args.cluster_num):
            if not args.tied:
                P_XZ = lambdas[i] * multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(data[n])
            else:
                P_XZ = lambdas[i] * multivariate_normal(mean=mus[i], cov=sigmas).pdf(data[n])
            Z_top.append(P_XZ)

        Z_top = np.array(Z_top)  # numerator which is P(x,z)
        Z_bottom = sum(Z_top)  # denominator which is P(x) =∑_k λ_k * N(X^(n)|μ_k,Σ_k)

        log_px.append(log(Z_bottom))

    log_px = np.array(log_px)
    ll = sum(log_px) / len(data)
    return ll


###
def main():
    import argparse
    import os
    print('Gaussian')  # Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true',
                        help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied', action='store_true',
                        help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print(
            'You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))


if __name__ == '__main__':
    main()
