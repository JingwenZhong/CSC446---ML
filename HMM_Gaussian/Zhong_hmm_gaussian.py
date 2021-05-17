#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4/21/2021

@author: Jingwen
"""

import numpy as np

if not __file__.endswith('_hmm_gaussian.py'):
    print(
        'ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs446/data/em/"  # TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)


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


def init_model(args):
    if args.cluster_num:
        # TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        mus = np.zeros((args.cluster_num, 2))
        # transitions = np.zeros((args.cluster_num,args.cluster_num))
        transitions = np.random.rand(args.cluster_num, args.cluster_num)  # size=(args.cluster_num*args.cluster_num)
        # transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num)  # probability for starting in each state

        for i in range(args.cluster_num):
            mus[i] = np.random.normal(0, 1, 2)  # use normal distribution with mean=0, sigma = 1.5
            transitions[i] = transitions[i] / sum(transitions[i])  # each line have to add up to 1
            initials[i] = 1 / args.cluster_num  # have to add up to 1

        if not args.tied:
            sigmas = np.zeros((args.cluster_num, 2, 2))
            for i in range(args.cluster_num):
                sigmas[i] = np.eye(2)
        else:
            sigmas = np.eye(2)  # array([[1., 0.],[0., 1.]])
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file, 'r') as f:
            for line in f:
                # each line is a cluster, and looks like this:
                # initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float, line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5], vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    # TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = {'initials': initials, 'transitions': transitions, 'mus': mus, 'sigmas': sigmas}

    return model


def extract_parameters(model):
    # TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model['initials']
    transitions = model['transitions']
    mus = model['mus']
    sigmas = model['sigmas']
    return initials, transitions, mus, sigmas


def emission(model, data, args):
    from scipy.stats import multivariate_normal

    _, _, mus, sigmas = extract_parameters(model)

    B = np.zeros((len(data), args.cluster_num))
    for n in range(len(data)):
        for i in range(args.cluster_num):
            if not args.tied:
                B[n, i] = multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(data[n])  # emission using gaussian
            else:
                B[n, i] = multivariate_normal(mean=mus[i], cov=sigmas).pdf(data[n])
    return B


def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data), args.cluster_num))
    log_likelihood = 0.0
    # TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    # NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value,
    # and increment log_likelihood by the log of the value you normalized by.
    # This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize
    # (you don't need to do anything different than what's in the notes).

    initials, transitions, mus, sigmas = extract_parameters(model)
    B = emission(model, data, args)  # emission

    for n in range(len(data)):
        if n == 0:  # initials modify first observation alpha[0,*], i.e., initials are P(y0= i |START)
            alphas[n] = initials * B[n]
        else:
            alphas[n] = np.dot(np.transpose(transitions), alphas[n - 1]) * B[n]  # ∑jα(n−1,j)⋅P(yn=i|yn−1=j)⋅P(Xn|yn=i)

        Z = np.sum(alphas[n])  # normalization
        alphas[n] = alphas[n] / Z

        log_likelihood += log(Z)

    return alphas, log_likelihood


def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data), args.cluster_num))
    # TODO: Calculate and return backward probabilities (normalized like in forward before)
    initials, transitions, mus, sigmas = extract_parameters(model)
    B = emission(model, data, args)  # emission
    betas[len(data) - 1, :] = 1

    for n in reversed(range(len(data) - 1)):
        betas[n] = np.dot((transitions * betas[n + 1]), B[n + 1])  # ∑jβ(n+1,j)⋅P(yn+1=j|yn=i)⋅P(Xn+1|yn+1=j)⋅
        betas[n] = betas[n] / sum(betas[n])  # normalization
    return betas


def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    # TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    likelihood_train = []
    likelihood_dev = []

    for iteration in range(args.iterations):

        initials, transitions, mus, sigmas = extract_parameters(model)
        B = emission(model, train_xs, args)  # emission

        # E step:
        alphas, _ = forward(model, train_xs, args)  # forward
        betas = backward(model, train_xs, args)  # backward

        # gamma
        gamma = alphas * betas
        for n in range(len(train_xs)):  # normalization gama
            gamma[n] = gamma[n] / sum(gamma[n])

        # Ksi
        ksi = np.zeros((len(train_xs), args.cluster_num, args.cluster_num))
        for n in range(len(train_xs)):
            if n != 0:
                ksi[n] = np.transpose(alphas[n - 1] * np.transpose(transitions * betas[n] * B[n]))
                ksi[n] = ksi[n]/np.sum(ksi[n])

        # M step:
        for i in range(args.cluster_num):
            mus[i] = np.dot(gamma[:, i], train_xs) / sum(gamma[:, i])  # update mus
            # update sigmas:
            if not args.tied:
                sigmas[i] = np.dot(gamma[:, i] * (train_xs - mus[i]).T, (train_xs - mus[i])) / sum(gamma[:, i])
            else:
                sigmas += np.dot(gamma[:, i] * (train_xs - mus[i]).T, (train_xs - mus[i]))

        if args.tied:
            sigmas = sigmas / len(train_xs)

        ec = np.zeros((args.cluster_num, args.cluster_num))  # expected count
        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                for n in range(len(train_xs)):
                    ec[i, j] += ksi[n, i, j]
            transitions[i] = ec[i] / sum(ec[i])  # update transitions

        initials = gamma[0]  # update initials

        # update model:
        model = {'initials': initials, 'transitions': transitions, 'mus': mus, 'sigmas': sigmas}

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
        plt.savefig("Zhong_hmm_gaussian_ll.jpg")
        plt.show()

    return model


def average_log_likelihood(model, data, args):
    # TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    # NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0

    _, log_likelihood = forward(model, data, args)
    ll = log_likelihood / len(data)

    return ll


def main():
    import argparse
    import os
    print('Gaussian')  # Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
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
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str, a))

        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '), transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '), mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '), map(lambda s: np.nditer(s), sigmas)))))


if __name__ == '__main__':
    main()
