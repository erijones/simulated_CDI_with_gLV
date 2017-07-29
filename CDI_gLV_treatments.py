#!/bin/python
#
# Version 0.9 
# This file generates figures for the paper 'In silico analysis of C. difficile
# infection: remediation techniques and biological adaptations' by Eric Jones
# and Jean Carlson. The code is still in a preliminary form. This code is
# covered under GNU GPLv3.
#
# Send questions to Eric Jones at ewj@physics.ucsb.edu
#
###############################################################################





# if running to see code performance
timing = True

import time
tic0 = time.clock()

import pickle
import math
import sys
import numpy as np
import scipy.integrate as integrate
from collections import namedtuple
if timing: print("time elapsed for modules is", time.clock()-tic0)


# import data from Stein paper for parameters and initial conditions
def import_data():
    # import messy variables
    with open('stein_parameters.csv','r') as f:
        var_data = [line.strip().split(",") for line in f]
    # import messy initial conditions
    with open('stein_ic.csv','r') as f:
        ic_data = [line.strip().split(",") for line in f]
    return var_data[1:], ic_data

# extract messy (i.e. string) parameters, cast as floats, and then numpy arrays
def parse_data(var_data, scenario='default'):
    # extract microbe labels, to be placed in legend
    labels = [label.replace("_"," ") for label in var_data[-1] if label.strip()]
    # extract M, mu, and eps from var_data
    str_inter   = [elem[1:(1+len(labels))] for elem in var_data][:-1]
    str_gro     = [elem[len(labels)+1] for elem in var_data][:-1]
    str_sus     = [elem[len(labels)+2] for elem in var_data][:-1]
    float_inter = [[float(value) for value in row] for row in str_inter]
    float_gro   = [float(value) for value in str_gro]
    float_sus   = [float(value) for value in str_sus]
    # convert to numpy arrays
    M   = np.array(float_inter)
    mu  = np.array(float_gro)
    eps = np.array(float_sus)
    # swap C. diff so that it is the last element
    c_diff_index                     = 8
    M[:, [c_diff_index, -1]]         = M[:, [-1, c_diff_index]]
    M[[c_diff_index, -1], :]         = M[[-1, c_diff_index], :]
    mu[c_diff_index], mu[-1]         = mu[-1], mu[c_diff_index]
    eps[c_diff_index], eps[-1]       = eps[-1], eps[c_diff_index]
    labels[c_diff_index], labels[-1] = labels[-1], labels[c_diff_index]
    c_diff_index                     = labels.index("Clostridium difficile")
    # return different sizes for different scenarios
    if scenario == 'default':
        return labels, mu, M, eps
    if scenario == 'sporulation':
        N       = len(mu)
        new_M   = np.zeros((N+1, N+1)); new_M[0:N, 0:N] = M
        new_mu  = np.append(mu, 0);
        new_eps = np.append(eps, 0)
        labels.append('Clostridium difficile spores')
        return labels, new_mu, new_M, new_eps
    if scenario in ('mutation_1', 'mutation_2'):
        N               = len(mu)
        new_M           = np.zeros((N+1, N+1));
        new_M[0:N, 0:N] = M
        # mutation incurs fitness cost
        fitness_penalty = .9
        new_mu          = np.append(mu, fitness_penalty*mu[c_diff_index])
        # make mutant antibiotic resistant
        new_eps         = np.append(eps, 0)
        labels.append('Mutant Clostridium difficile')
        new_M[:, N] = np.append(M[:, c_diff_index], 0)
        # two different mutation regimes:
        if scenario == 'mutation_1':
            new_M[N, :] = fitness_penalty * np.append(M[c_diff_index, :], M[c_diff_index, c_diff_index])
        if scenario == 'mutation_2':
            new_M[N, :] = np.append(M[c_diff_index, :], M[c_diff_index, c_diff_index])
            new_M[c_diff_index, N] = M[c_diff_index, c_diff_index] # M_cm = M_cc
            new_M[N, c_diff_index] = -.02 #M_mc
            new_M[N, N] = -.06 #M_mm
        return labels, new_mu, new_M, new_eps

# extract messy (i.e. string) ICs, cast as floats, and then numpy arrays
def parse_ic(ic_vars, scenario='default'):
    ic_data, ic_num      = ic_vars
    if ic_num == 'inf_ss':
        scn = Scenario(scenario            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , False ,
                       'save'               , 'basic_traj_log' , 'new_figs/throwaway.pdf')
        t, y = master(scn)
        return y[-1]
    else:
        ic_list_str          = [[elem[i] for elem in ic_data][5:-2] for i in \
                                 range(1,np.shape(ic_data)[1]) if float(ic_data[3][i])==0]
        ic_list_float        = [[float(value) for value in row] for row in ic_list_str]
        #[print(num, sum(ic_list_float[:][num])) for num in range(9)]
        ic                   = np.array(ic_list_float[:][ic_num])
        old_c_diff_index     = 8
        c_diff_index         = labels.index("Clostridium difficile")
        ic[c_diff_index], ic[old_c_diff_index] = ic[old_c_diff_index], ic[c_diff_index]

        #if ic_num == 7:
        #    ic[-1] = 0.0
        if scenario == 'default':
            return ic
        elif scenario in ('sporulation', 'mutation_1', 'mutation_2'):
            new_ic = np.append(ic, 0)
            return new_ic
        else:
            raise ValueError('invalid scenario type')

def master_solve(ic, sc):
    if sc.solve_type == 'microbe_trajectory':

        if sc.transplant_time_1 < sc.transplant_time_2:
            t01 = np.linspace(0, sc.transplant_time_1, num=1001)
            t12 = np.linspace(sc.transplant_time_1, sc.transplant_time_2, num=1001)
            t23 = np.linspace(sc.transplant_time_2, sc.t_end, num=1001)
            treatment_params = [sc.treatment_type, sc.treatment_dose, sc.treatment_duration]

            y01                = integrate.odeint(master_integrand, ic, t01, \
                                                  args=(sc.scenario_type, treatment_params), atol = 1e-12)
            transplant_1       = m_generate_sample(sc.transplant_type_1, sc.transplant_strength_1, sc.scenario_type)
            after_transplant_1 = y01[-1] + transplant_1
            y12                = integrate.odeint(master_integrand, after_transplant_1, t12, \
                                                  args=(sc.scenario_type, treatment_params), atol = 1e-12)
            transplant_2       = m_generate_sample(sc.transplant_type_2, sc.transplant_strength_2, sc.scenario_type)
            after_transplant_2 = y12[-1] + transplant_2
            y23                = integrate.odeint(master_integrand, after_transplant_2, t23, \
                                                  args=(sc.scenario_type, treatment_params), atol=1e-12)
        else:
            t01 = np.linspace(0, sc.transplant_time_2, num=1001)
            t12 = np.linspace(sc.transplant_time_2, sc.transplant_time_1, num=1001)
            t23 = np.linspace(sc.transplant_time_1, sc.t_end, num=1001)
            treatment_params = [sc.treatment_type, sc.treatment_dose, sc.treatment_duration]

            y01                = integrate.odeint(master_integrand, ic, t01, \
                                                  args=(sc.scenario_type, treatment_params), atol = 1e-12)
            transplant_1       = m_generate_sample(sc.transplant_type_2, sc.transplant_strength_2, sc.scenario_type)
            after_transplant_1 = y01[-1] + transplant_1
            y12                = integrate.odeint(master_integrand, after_transplant_1, t12, \
                                                  args=(sc.scenario_type, treatment_params), atol = 1e-12)
            transplant_2       = m_generate_sample(sc.transplant_type_1, sc.transplant_strength_1, sc.scenario_type)
            after_transplant_2 = y12[-1] + transplant_2
            y23                = integrate.odeint(master_integrand, after_transplant_2, t23, \
                                                  args=(sc.scenario_type, treatment_params), atol=1e-12)
        return np.concatenate((t01,t12,t23)), np.vstack((y01,y12,y23))
    if sc.solve_type in ('ss_u_transplant', 'ss_duration_dose'):
        # xx represents u, yy represents transplant
        if sc.solve_type=='ss_u_transplant':
            nx, ny = (21, 3)
            xx = np.linspace(0, 2.0, nx)
            yy = np.linspace(0, 1.0, ny)
        if sc.solve_type=='ss_duration_dose' or sc.filename =='new_figs/fig_3a.pdf':
            #nx, ny = (49, 41)
            nx, ny = (33, 21)
            #nx, ny = (10,10)
            xx = np.linspace(0, 2.0, nx)
            yy = np.linspace(0, 1.0, ny)
        xv, yv = np.meshgrid(xx, yy)
        z = np.array([[0.0 for x in range(len(xx))] for y in range(len(yy))])
        for (x_i,x) in enumerate(xx):
            for (y_i,y) in enumerate(yy):
                u_density = x; transplant_strength = y
                if sc.solve_type == 'ss_u_transplant':
                    temp_scn = Scenario('default'       , sc.ic_num            ,
                                   sc.transplant_type_1 , sc.transplant_time_1 , transplant_strength      ,
                                   sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                                   sc.treatment_type    , u_density            , sc.treatment_duration    ,
                                   'microbe_trajectory' , sc.t_end             , False                    ,
                                   'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                if sc.solve_type == 'ss_duration_dose':
                    if '8' in sc.filename:
                        #print('multiplying transplant strength')
                        temp_scn = Scenario('default'       , sc.ic_num            ,
                                   sc.transplant_type_1 , sc.transplant_time_1 , sc.transplant_strength_1,
                                   sc.transplant_type_2 , sc.transplant_time_2 , 4*transplant_strength,
                                   sc.treatment_type    , u_density            , sc.treatment_duration    ,
                                   'microbe_trajectory' , sc.t_end             , False                    ,
                                   'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                    else:
                        temp_scn = Scenario('default'       , sc.ic_num            ,
                                   sc.transplant_type_1 , sc.transplant_time_1 , sc.transplant_strength_1,
                                   sc.transplant_type_2 , sc.transplant_time_2 , transplant_strength,
                                   sc.treatment_type    , u_density            , sc.treatment_duration    ,
                                   'microbe_trajectory' , sc.t_end             , False                    ,
                                   'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master_solve(ic, temp_scn)

                # calculate the shannon diversity, effectively hashing the
                # steady state
                y_total = np.array([sum(y[i]) for i in range(len(t))])
                biodiv =  [-sum([(y[i,j]/y_total[i])*math.log10(y[i,j]/y_total[i]) \
                                 if ((y[i,j] > 0.001) and (y_total[i] > 0)) else 0 \
                                 for j in range(len(y[0]))]) for i in range(len(t))]
                z[y_i, x_i] = biodiv[-1]
        return (xv, yv), z

# solve for microbe trajectories; uses integrand() and u()
def solve(solve_vars, solve_type='default'):
    if ((solve_type == 'default') or (solve_type == 'sporulation') or (solve_type == 'mutation')):
        ic, *params, t_end = solve_vars
        t = np.linspace(0, t_end, num=1000)
        y = integrate.odeint(integrand, ic, t, args=(*params, solve_type), atol=1e-12)
        return t, y
    if solve_type == 'transplant':
        ic, *params, t_end, sample, transplant_time = solve_vars
        t1 = np.linspace(0, transplant_time, num=1000)
        t2 = np.linspace(transplant_time, t_end, num=1000)
        y1 = integrate.odeint(integrand, ic, t1, args=(*params, solve_type), atol=1e-12)
        y2 = integrate.odeint(integrand, y1[-1] + sample, t2, args=(*params, solve_type), atol=1e-12)
        return np.concatenate((t1,t2)), np.vstack((y1,y2))
    if solve_type == 'transplant_spor':
        ic, *params, t_end, sample, transplant_time = solve_vars
        t1 = np.linspace(0, transplant_time, num=1000)
        t2 = np.linspace(transplant_time, t_end, num=1000)
        y1 = integrate.odeint(integrand, ic, t1, args=(*params, 'sporulation'), atol=1e-15)
        y2 = integrate.odeint(integrand, y1[-1,:] + sample, t2, args=(*params, 'sporulation'), atol=1e-15)
        return np.concatenate((t1,t2)), np.vstack((y1,y2))
    if solve_type == 'ss_basins':
        ic, *params, t_end, sample_vars, transplant_time = solve_vars
        mu, M, eps, u_params = params
        transplant_time, strength, labels, sample_of = sample_vars
        a, b, u_type = u_params
        print('u_type is', u_type)
        nx, ny = (41, 41)
        xx = np.linspace(0, 2, nx)
        yy = np.linspace(0, 8, ny)
        xv, yv = np.meshgrid(xx, yy)
        z = np.array([[0.0 for x in range(len(xx))] for y in range(len(yy))])
        new_sample_vars = (transplant_time, strength, labels, sample_of)
        new_sample = generate_sample(new_sample_vars, sample_type='c_diff')
        ic = ic + new_sample
        for (x_i,x) in enumerate(xx):
            for (y_i,y) in enumerate(yy):
                u_density = x; duration = y
                if duration == 0:
                    new_u_params = [u_density*duration, 1, u_type]
                else:
                    new_u_params = [u_density*duration, duration, u_type]
                new_params = mu, M, eps, new_u_params
                solve_vars = (ic, *new_params, t_end) #, new_sample, transplant_time)
                ##u_density = x; transplant_strength = y
                ##new_u_params = [u_density, 10, u_type]
                ##new_params = mu, M, eps, new_u_params
                ##new_sample_vars = (transplant_time, transplant_strength, labels, sample_of)
                ##new_sample = generate_sample(new_sample_vars, sample_type='ic_plus_cdiff_spor')
                ##solve_vars = (ic, *new_params, t_end, new_sample, transplant_time)
                #u_density = x; sample_strength = y
                #new_u_params = [u_density, 1, 'default']
                #new_params = mu, M, eps, new_u_params
                #new_sample_vars = (transplant_time, sample_strength, labels)
                #new_sample = generate_sample(new_sample_vars, sample_type='c_diff')
                #solve_vars = (ic, *new_params, t_end, new_sample, transplant_time)
                ##t, y = solve(solve_vars, 'transplant_spor')
                t, y = solve(solve_vars, 'default')

                y_total = np.array([sum(y[i]) for i in range(len(t))])
                #print(y[1896, 4], y_total[1896])
                biodiv = [-sum([(y[i,j]/y_total[i])*math.log10(y[i,j]/y_total[i]) if \
                          ((y[i,j] > 0.001) and (y_total[i] > 0)) else 0 for j
                          in range(len(y[0]))]) for i in range(len(t))]
                #biodiv = [-sum([(y[i,j]/y_total[i]) if \
                #          (abs(y[i,j]) > 0.001) else 0 for j in range(len(y[0]))]) for i in range(len(t))]

                z[y_i, x_i] = biodiv[-1]
        return (xv, yv), z
    if solve_type == 'ss_basins_u_cdiff':
        ic, *params, t_end, sample_vars, transplant_time = solve_vars
        mu, M, eps, u_params = params
        transplant_time, strength, labels, sample_of = sample_vars
        a, b, u_type = u_params
        print('u_type is', u_type)
        nx, ny = (11, 11)
        xx = np.linspace(0, 2, nx)
        yy = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(xx, yy)
        z = np.array([[0.0 for x in range(len(xx))] for y in range(len(yy))])
        #new_sample_vars = (transplant_time, strength, labels, sample_of)
        #new_sample = generate_sample(new_sample_vars, sample_type='c_diff')
        #ic = ic + new_sample
        for (x_i,x) in enumerate(xx):
            for (y_i,y) in enumerate(yy):
                ##u_density = x; strength = y
                ##new_u_params = [u_density, 1, u_type]
                ##new_params = mu, M, eps, new_u_params
                ##ic[8] = strength
                ##solve_vars = (ic, *new_params, t_end) #, new_sample, transplant_time)

                u_density = x; transplant_strength = y
                new_u_params = [u_density, 10, u_type]
                new_params = mu, M, eps, new_u_params
                new_sample_vars = (transplant_time, transplant_strength, labels, sample_of)
                new_sample = generate_sample(new_sample_vars, sample_type='c_diff')
                solve_vars = (ic, *new_params, t_end, new_sample, transplant_time)
                #u_density = x; sample_strength = y
                #new_u_params = [u_density, 1, 'default']
                #new_params = mu, M, eps, new_u_params
                #new_sample_vars = (transplant_time, sample_strength, labels)
                #new_sample = generate_sample(new_sample_vars, sample_type='c_diff')
                #solve_vars = (ic, *new_params, t_end, new_sample, transplant_time)
                t, y = solve(solve_vars, 'transplant_spor')
                #t, y = solve(solve_vars, 'default')

                y_total = np.array([sum(y[i]) for i in range(len(t))])
                #print(y[1896, 4], y_total[1896])
                biodiv = [-sum([(y[i,j]/y_total[i])*math.log10(y[i,j]/y_total[i]) if \
                          ((y[i,j] > 0.001) and (y_total[i] > 0)) else 0 for j
                          in range(len(y[0]))]) for i in range(len(t))]
                #biodiv = [-sum([(y[i,j]/y_total[i]) if \
                #          (abs(y[i,j]) > 0.001) else 0 for j in range(len(y[0]))]) for i in range(len(t))]

                z[y_i, x_i] = biodiv[-1]
        return (xv, yv), z
    if solve_type == 'ss_basins_delay':
        ic, *params, t_end, sample_vars, transplant_time = solve_vars
        mu, M, eps, u_params = params
        old_dose, old_duration, old_u_type = u_params
        if len(mu) == 11:
            temp_type = 'default'
        elif len(mu) == 12:
            temp_type = 'sporulation'
        transplant_time, strength, labels, sample_of = sample_vars
        nx, ny = (21, 21)
        #xx = np.linspace(0, 2, nx)
        xx = np.linspace(0, 2, nx)
        yy = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(xx, yy)
        z = np.array([[0.0 for x in range(len(xx))] for y in range(len(yy))])
        for (x_i,x) in enumerate(xx):
            for (y_i,y) in enumerate(yy):
                u_density = x; transplant_strength = y
                new_u_params = [u_density, 1, old_u_type]
                new_params = mu, M, eps, new_u_params
                first_transplant_time = 0.1
                first_end_time = .11
                new_sample_vars_first = (first_transplant_time,
                        transplant_strength, labels, sample_of)
                new_sample_first = generate_sample(new_sample_vars_first, sample_type='c_diff')
                solve_vars_first = (ic, *new_params, first_end_time, new_sample_first, first_transplant_time)
                if temp_type == 'sporulation':
                    t1, y1 = solve(solve_vars_first, 'transplant_spor')
                else:
                    t1, y1 = solve(solve_vars_first, 'transplant')

                ic_second = y1[-1]
                new_u_params_second = [u_density, 1-first_end_time, old_u_type]
                new_params_second = mu, M, eps, new_u_params_second
                second_transplant_time = 5
                new_sample_vars_second = (second_transplant_time-first_end_time,
                        transplant_strength, labels, sample_of)
                if temp_type == 'sporulation':
                    new_sample_second = \
                    np.append(generate_sample(new_sample_vars_second, \
                        sample_type='ic'),0)
                else:
                    new_sample_second = generate_sample(new_sample_vars_second, sample_type='ic')
                solve_vars_second = (ic_second, *new_params_second,
                        t_end-first_end_time, new_sample_second,
                        second_transplant_time-first_end_time)
                if temp_type == 'sporulation':
                    t2, y2 = solve(solve_vars_second, 'transplant_spor')
                else:
                    t2, y2 = solve(solve_vars_second, 'transplant')

                adjusted_t2 = t2 + t1[-1]
                t = np.concatenate((t1,adjusted_t2))
                y = np.vstack((y1,y2))

                #new_sample_vars_first = (transplant_time, transplant_strength, labels, sample_of)
                #u_density = x; sample_strength = y
                #new_u_params = [u_density, 1, 'default']
                #new_params = mu, M, eps, new_u_params
                #new_sample_vars = (transplant_time, sample_strength, labels)
                #new_sample = generate_sample(new_sample_vars, sample_type='c_diff')
                #solve_vars = (ic, *new_params, t_end, new_sample, transplant_time)

                y_total = np.array([sum(y[i]) for i in range(len(t))])
                #print(y[1896, 4], y_total[1896])
                biodiv = [-sum([(y[i,j]/y_total[i])*math.log10(y[i,j]/y_total[i]) if \
                          ((y[i,j] > 0.001) and (y_total[i] > 0)) else 0 for j in range(len(y[0]))]) for i in range(len(t))]
                #biodiv = [-sum([(y[i,j]/y_total[i]) if \
                #          (abs(y[i,j]) > 0.001) else 0 for j in range(len(y[0]))]) for i in range(len(t))]

                z[y_i, x_i] = biodiv[-1]
        return (xv, yv), z
    if solve_type == 'none':
        return True, True
    if solve_type == 'end_composition':
        return True, True

def master_integrand(Y, t, scenario, u_params):
    if (scenario == 'default'):
        return np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, u_params)*np.dot(np.diag(eps), Y)
    if (scenario == 'mutation_1'):
        c_diff_index = labels.index("Clostridium difficile"); k=0.002

        cd_term = np.zeros(len(mu)); xm_term = np.zeros(len(mu))
        cd_term[c_diff_index] = -k*Y[c_diff_index]*u(t, u_params)
        xm_term[-1] = k*Y[c_diff_index]*u(t, u_params)
        return np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, u_params)*np.dot(np.diag(eps), Y) + cd_term + xm_term
    if (scenario == 'mutation_2'):
        c_diff_index = labels.index("Clostridium difficile"); k=0.0005

        cd_term = np.zeros(len(mu)); xm_term = np.zeros(len(mu))
        cd_term[c_diff_index] = -k*Y[c_diff_index]*u(t, u_params)
        xm_term[-1] = k*Y[c_diff_index]*u(t, u_params)
        return np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, u_params)*np.dot(np.diag(eps), Y) + cd_term + xm_term
    if (scenario == 'sporulation'):
        c_diff_index = labels.index("Clostridium difficile");
        alpha = 0.31272*(1) # sporulation rate
        beta = alpha      # vegetation rate
        u_star = .05        # sporulation threshold

        cd_term = np.zeros(len(mu)); s_term = np.zeros(len(mu))
        cd_term[c_diff_index] = beta*Y[-1]*( u(t, u_params) < u_star )
        s_term[-1] = alpha*Y[c_diff_index]*( u(t, u_params) >= u_star ) \
                     - beta*Y[-1]*( u(t, u_params) < u_star )
        return np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, u_params)*np.dot(np.diag(eps), Y) + cd_term + s_term

# generalized Lotka-Volterra matrix ODE, for use in solve()
def integrand(Y, t, mu, M, eps, u_params=[1, 1, 'default'], integrand_type='default'):
    if (integrand_type == 'default') or (integrand_type == 'transplant'):
        return np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, *u_params)*np.dot(np.diag(eps), Y)
    if (integrand_type == 'sporulation'):
        c_diff_index = 10; alpha = 0.31272*(0); beta = 4*alpha; gamma = 0.0 # guesses
        # 0.31272 is eps for c diff
        normal_integrand = np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, *u_params)*np.dot(np.diag(eps), Y)
        u_cutoff = .05
        spor_stuff = np.zeros(len(mu))
        spor_stuff[c_diff_index] = beta*Y[-1]*(u(t, *u_params) < u_cutoff)
        spor_stuff[-1] = alpha*Y[c_diff_index]*(u(t, *u_params) >= u_cutoff) \
                         - beta*Y[-1]*(u(t, *u_params) < u_cutoff) - gamma*Y[-1]
        return normal_integrand + spor_stuff
    if (integrand_type == 'mutation'):
        c_diff_index = 10; k=0.002
        # 0.31272 is eps for c diff
        normal_integrand = np.dot(np.diag(mu), Y) + np.dot( np.diag(np.dot(M, Y)), Y) + \
                   u(t, *u_params)*np.dot(np.diag(eps), Y)

        mutation_term = np.zeros(len(mu))
        mutation_term[c_diff_index] = -k*Y[c_diff_index]*(u(t, *u_params) > 0)
        mutation_term[-1] = k*Y[c_diff_index]*(u(t, *u_params) > 0)
        return normal_integrand + mutation_term

# time-dependent antibiotic administration
def u(t, u_params):
    u_type, total_dose, duration = u_params
    if any(uu in u_type for uu in ('default', 'constant')):
        if total_dose==0:
            return 0
        else:
            return (total_dose/duration)*(t<duration)
    if u_type == 'none':
        return 0
    if u_type == 'tapered':
        return (t<duration)*( (total_dose/duration)*2 - \
                              2*(total_dose/(duration*duration))*t)
    if 'pulse' in u_type:
        return ((np.floor(t) % 3) == 0)*(total_dose/math.ceil(duration/3)) \
                                       *(t<duration)

def m_generate_sample(transplant_type, transplant_strength, scenario):
    if scenario == 'default':
        sample = np.zeros(11)
    else:
        sample = np.zeros(12)

    if transplant_type == 'none':
        return sample
    elif transplant_type == 'cdiff':
        c_diff_index = labels.index("Clostridium difficile")
        sample[c_diff_index] = 1000*(10**(-13)) # injected with 1000 c. diff
        return sample*transplant_strength
    elif isinstance(transplant_type, int):
        sample = parse_ic((ic_data, transplant_type), scenario)
        return sample*transplant_strength

# create "sample" to be injected into ode at transplant_time
def generate_sample(sample_vars, sample_type):
    if (sample_type == 'c_diff'):
        transplant_time, strength, labels, sample_of = sample_vars
        c_diff_index = labels.index("Clostridium difficile")
        sample = np.zeros(len(labels))
        sample[c_diff_index] = 1000*(10**(-13)) # injected with 1000 c. diff
        #sample[c_diff_index] = 10**(-2) # injected with 1000 c. diff
        return sample*strength
    if (sample_type == 'ic'):
        transplant_time, strength, labels, sample_of = sample_vars
        var_data, ic_data = import_data()
        new_ic_condition = np.zeros(len(labels))
        new_ic_condition = parse_ic((ic_data, sample_of), 'default')
        return new_ic_condition*strength
    if (sample_type == 'ic_plus_cdiff'):
        transplant_time, strength, labels, sample_of = sample_vars
        var_data, ic_data = import_data()
        new_ic_condition = np.zeros(len(labels))
        temp = parse_ic((ic_data, sample_of), 'default')
        new_ic_condition[0:len(temp)] = temp
        c_diff_index = labels.index("Clostridium difficile")
        c_diff_sample = np.zeros(len(labels))
        c_diff_sample[c_diff_index] = 1000*(10**(-13)) # injected with 1000 c. diff
        total_sample = strength*(new_ic_condition + c_diff_sample)
        return total_sample
    if (sample_type == 'ic_plus_cdiff_spor'):
        transplant_time, strength, labels, sample_of = sample_vars
        var_data, ic_data = import_data()
        new_ic_condition = np.zeros(len(labels))
        temp = parse_ic((ic_data, sample_of), 'default')
        new_ic_condition[0:len(temp)] = temp
        c_diff_index = labels.index("Clostridium difficile")
        c_diff_sample = np.zeros(len(labels))
        c_diff_sample[c_diff_index] = 1000*(10**(-13)) # injected with 1000 c. diff
        total_sample = strength*(new_ic_condition + c_diff_sample)
        return total_sample

def make_bw(plot_data):
    import matplotlib.pyplot as plt
    (xv, yv), z = plot_data
    diff1 = np.zeros(np.shape(z))
    for i1, row in enumerate(z):
        current = row[0]
        for i2, item in enumerate(row):
            if abs(item - current) > .001:
                diff1[i1][i2] = 1
            current = item
    diff2 = np.zeros(np.shape(z))
    for i1, column in enumerate(z.T):
        current = column[0]
        for i2, item in enumerate(column):
            if abs(item - current) > .001:
                diff2[i2][i1] = 1
            current = item
    print(diff1)
    print(xv)
    print(yv)

    for i, row in enumerate(diff1):
        for j, item in enumerate(row):
            if item == 1:
                assert diff1[i,j] == item
                try:
                    plt.plot([xv[0][j], xv[0][j]], [yv[i][0], yv[i+1][0]], color='k')
                except IndexError:
                    print("out of bounds")
    for i, row in enumerate(diff2):
        for j, item in enumerate(row):
            if item == 1:
                assert diff2[i,j] == item
                try:
                    plt.plot([xv[0][j], xv[0][j+1]], [yv[i][0], yv[i][0]], color='k')
                except IndexError:
                    print("out of bounds")
    plt.axis([0, 2, 0, 1])
    plt.show()

def m_plot(plot_data, sc):
    # master plot
    if sc.if_plot == 'none':
        print('suppressing', sc.plot_type, 'plot for scenario', sc.scenario_type)
        return
    import matplotlib.pyplot as plt
    import seaborn.apionly as sns
    #plt.style.use('seaborn-muted')
    part_1 = sns.color_palette("Set1", n_colors = 8)
    part_2 = sns.color_palette("Set2", n_colors = 8)
    palette = part_1[1:] + part_2
    sns.set_palette(palette)
    plt.clf()
    # plots log(total microbe composition) (y-axis) over time (x-axis), where at
    # each time point microbe populations are plotted proportional in different
    # colors. C. diff is red. 
    if sc.plot_type in ('basic_traj_log'):
        t, y = plot_data;

        y_total       = np.array([sum(y[i]) for i in range(len(t))])
        y_log_total   = 11 + np.log10(y_total) # +11 to compensate for microbe units
        y_log_species = np.array( [(y[i,:]/y_total[i])*y_log_total[i] \
                                   for i in range(len(t))] )
        # build layered microbe composition
        layered_y      = np.zeros(np.shape(y))
        layered_y[:,0] = y_log_species[:,0]
        for i in range(1,len(labels)):
            layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]

        # configure microbe coloring in plot
        fig, ax = plt.subplots()
        c_diff_index = labels.index("Clostridium difficile")
        for i in range(len(labels)):
            # remember colors
            if i == c_diff_index:
                color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='r')
            elif i == (1+c_diff_index):
                color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='k')
            else:
                color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0)

            # fill in colors
            if i == 0:
                # rasterized to make pdfs smaller
                ax.fill_between(t, 0, layered_y[:,0], rasterized=True, linewidth=0)
            else:
                ax.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
                                facecolor=color_wheel.get_color(), rasterized=True, \
                                linewidth=0)
        # set axes labels
        ax.set_xlabel('$t$ (days)')
        ax.set_ylabel('log$_{10}$ microbe concentration')
        ax.set_xlim([0, t[-1]])
        ax.set_ylim([0, 1.05*max(y_log_total)])
        plt.tight_layout()

        if sc.if_legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height*0.1, box.width,
                box.height*0.9])
            leg = plt.legend(labels, prop={'size':6}, loc='upper center', ncol=3,
                             bbox_to_anchor=(0.45, -0.2), fancybox=True)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(3.0)

    if sc.plot_type in ('4_traj_plot'):
        print('here!')
        (t1, y1), (t2, y2), (t3, y3), (t4, y4) = plot_data;
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        for (index, (t, y)) in enumerate([(t1, y1), (t2, y2), (t3, y3), (t4, y4)]):
            if index == 0: ax = ax1
            elif index == 1: ax = ax2
            elif index == 2: ax = ax3
            elif index == 3: ax = ax4

            y_total       = np.array([sum(y[i]) for i in range(len(t))])
            y_log_total   = 11 + np.log10(y_total) # +11 to compensate for microbe units
            y_log_species = np.array( [(y[i,:]/y_total[i])*y_log_total[i] \
                                       for i in range(len(t))] )
            # build layered microbe composition
            layered_y      = np.zeros(np.shape(y))
            layered_y[:,0] = y_log_species[:,0]
            for i in range(1,len(y[0])):
                layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]

            # configure microbe coloring in plot
            c_diff_index = labels.index("Clostridium difficile")
            for i in range(len(y[0])):
                # remember colors
                if i == c_diff_index:
                    color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='r')
                elif i == (1+c_diff_index):
                    color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='k')
                else:
                    color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0)

                # fill in colors
                if i == 0:
                    # rasterized to make pdfs smaller
                    ax.fill_between(t, 0, layered_y[:,0], rasterized=True, linewidth=0)
                else:
                    ax.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
                                    facecolor=color_wheel.get_color(), rasterized=True, \
                                    linewidth=0)
            # set axes labels
            if index == 2 or index == 3:
                ax.set_xlabel('$t$ (days)')
            if index == 0 or index == 2:
                ax.set_ylabel('log(\# microbes)')
            if sc.scenario_type == 'sporulation':
                if index == 0: ax.text(-3, 12, '\\textbf{(a)}', ha='left')
                if index == 1: ax.text(-3, 12, '\\textbf{(b)}', ha='left')
                if index == 2: ax.text(-3, 12, '\\textbf{(c)}', ha='left')
                if index == 3: ax.text(-3, 12, '\\textbf{(d)}', ha='left')
            else:
                if index == 0: ax.text(-23, 13, '\\textbf{(a)}', ha='left')
                if index == 1: ax.text(-23, 13, '\\textbf{(b)}', ha='left')
                if index == 2: ax.text(-23, 13, '\\textbf{(c)}', ha='left')
                if index == 3: ax.text(-23, 13, '\\textbf{(d)}', ha='left')
                if index == 0: ax.text(140, 13.0, 'control', ha='right', fontsize=14)
                if index == 1: ax.text(140, 13.0, 'high antibiotic', ha='right', fontsize=14)
                if index == 2: ax.text(140, 13.0, 'low antibiotic + CD', ha='right', fontsize=14)
                if index == 3: ax.text(140, 13.0, 'high antibiotic + CD', ha='right', fontsize=14)
                if index == 3: ax.text(145, 10, '\\textbf{CD}', ha='right',
                        color='white', fontsize=20)
            ax.set_xlim([0, t[-1]])
            #ax.set_ylim([0, 1.05*max(y_log_total)])
            ax.set_ylim([0, 14.5])
        plt.tight_layout(pad=.3)

    if sc.plot_type in ('2_traj_plot'):
        print('here!')
        (t1, y1), (t2, y2)  = plot_data;
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6.4,3.2))
        for (index, (t, y)) in enumerate([(t1, y1), (t2, y2)]):
            if index == 0: ax = ax1
            elif index == 1: ax = ax2

            y_total       = np.array([sum(y[i]) for i in range(len(t))])
            y_log_total   = 11 + np.log10(y_total) # +11 to compensate for microbe units
            y_log_species = np.array( [(y[i,:]/y_total[i])*y_log_total[i] \
                                       for i in range(len(t))] )
            # build layered microbe composition
            layered_y      = np.zeros(np.shape(y))
            layered_y[:,0] = y_log_species[:,0]
            for i in range(1,len(y[0])):
                layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]

            # configure microbe coloring in plot
            c_diff_index = labels.index("Clostridium difficile")
            for i in range(len(y[0])):
                # remember colors
                if i == c_diff_index:
                    color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='r')
                elif i == (1+c_diff_index):
                    if sc.scenario_type == 'mutation_2':
                        color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='grey')
                    else:
                        color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0, color='k')
                else:
                    color_wheel, = ax.plot(t, layered_y[:,i], linewidth=0.0)

                # fill in colors
                if i == 0:
                    # rasterized to make pdfs smaller
                    ax.fill_between(t, 0, layered_y[:,0], rasterized=True, linewidth=0)
                else:
                    ax.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
                                    facecolor=color_wheel.get_color(), rasterized=True, \
                                    linewidth=0)
            # set axes labels
            ax.set_xlabel('$t$ (days)')

            params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
            plt.rcParams.update(params)

            if index == 0:
                ax.set_ylabel('log(\# microbes)')
            if index == 0: ax.text(-16, 13, '\\textbf{(a)}', ha='left')
            if index == 1: ax.text(-16, 13, '\\textbf{(b)}', ha='left')
            if sc.scenario_type == 'mutation_2':
                throwaway = 0
            else:
                if index == 0: ax.text(97, 13.0, r'day 1 transplant $\Rightarrow$ no CDI', ha='right', fontsize=14)
                if index == 1: ax.text(97, 13.0, r'day 7 transplant $\Rightarrow$ CDI', ha='right', fontsize=14)
                if index == 1: ax.text(95, 10, '\\textbf{CD}', ha='right',
                        color='white', fontsize=20)
            ax.set_xlim([0, t[-1]])
            #ax.set_ylim([0, 1.05*max(y_log_total)])
            ax.set_ylim([0, 14.5])
            ax.set_yticks([0, 5, 10])
            ax.set_yticklabels([0, 5, 10])
            plt.tight_layout(pad=.3)

    if sc.plot_type in ('basic_legend'):
        t, y = plot_data;

        y_total       = np.array([sum(y[i]) for i in range(len(t))])
        y_log_total   = 11 + np.log10(y_total) # +11 to compensate for microbe units
        y_log_species = np.array( [(y[i,:]/y_total[i])*y_log_total[i] \
                                   for i in range(len(t))] )
        # build layered microbe composition
        layered_y      = np.zeros(np.shape(y))
        layered_y[:,0] = y_log_species[:,0]
        for i in range(1,len(labels)):
            layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]

        # configure microbe coloring in plot
        fig, ax = plt.subplots()
        c_diff_index = labels.index("Clostridium difficile")
        if sc.scenario_type.startswith('mutation'):
            for i in range(len(labels)):
                # remember colors
                if i == c_diff_index:
                    color_wheel, = ax.plot([0], [0], linewidth=3.0, color='r')
                elif i == (1+c_diff_index):
                    color_wheel, = ax.plot([0], [0], linewidth=3.0, color='grey')
                else:
                    color_wheel, = ax.plot([0], [0], linewidth=3.0)
        else:
            for i in range(len(labels)):
                # remember colors
                if i == c_diff_index:
                    color_wheel, = ax.plot([0], [0], linewidth=3.0, color='r')
                elif i == (1+c_diff_index):
                    color_wheel, = ax.plot([0], [0], linewidth=3.0, color='k')
                else:
                    color_wheel, = ax.plot([0], [0], linewidth=3.0)
        ax.set_axis_off()
        ax.legend(labels, prop={'size':8}, loc='center', ncol=2,
                frameon=False)

    if sc.plot_type in ('food_web'):
        #print(M)

        # configure microbe coloring in plot
        fig, ax = plt.subplots()
        c_diff_index = labels.index("Clostridium difficile")
        wheels = []
        for i in range(len(labels)):
            # remember colors
            if i == c_diff_index:
                wheels.append(ax.plot([0], [0], linewidth=0.0, color='r'))
            elif i == (1+c_diff_index):
                wheels.append(ax.plot([0], [0], linewidth=0.0, color='k'))
            else:
                wheels.append(ax.plot([0], [0], linewidth=0.0))

        import matplotlib.patches as mpatches
        circ_size = 0.15
        circs = []; xx = []; yy = [];
        print('colorwheel is', wheels[1][0].get_color())
        for i in range(len(M)-1):
            xx.append(np.sin(2*np.pi*i/(len(M)-1)))
            yy.append(np.cos(2*np.pi*i/(len(M)-1)))
            circs.append(plt.Circle((xx[-1], yy[-1]), circ_size, lw=0,
                         color=wheels[i][0].get_color()))
        xx.append(0); yy.append(0)
        circs.append(plt.Circle((0, 0), circ_size, lw=0, color=wheels[-1][0].get_color()))
        for i in range(len(M)):
            for j in range(len(M)):
                if i != j:
                    cutoff = 2
                    if M[i][j] > 0:
                        arrow_color = 'green'
                    else:
                        arrow_color = 'red'
                    if (i==c_diff_index) or (j==c_diff_index):
                        curve_type = 'arc3,rad=0.2'
                    elif (( (i-j) % (len(M)-1)) > cutoff) and (((i-j) % (len(M)-1)) > cutoff):
                        curve_type = 'arc3,rad=0.2'
                    else:
                        curve_type = 'arc3,rad=-0.2'
                    ax.annotate("", xy=(xx[i],yy[i]), xytext=(xx[j],yy[j]),
                            zorder=1,
                            arrowprops = dict(arrowstyle='->', facecolor=arrow_color,
                                edgecolor=arrow_color, alpha=abs(M[i][j]),
                            patchA=mpatches.Circle((xx[i], yy[i]), circ_size), shrinkA=25,
                            patchB=mpatches.Circle((xx[j], yy[j]), circ_size), shrinkB=20,
                            connectionstyle=curve_type, linewidth = 2*abs(M[i][j])))

        for i in range(len(M)):
            ax.add_artist(circs[i])



        edge = (1+circ_size)*1.02
        plt.axis([-edge, edge, -edge, edge])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
            ## fill in colors
            #if i == 0:
            #    # rasterized to make pdfs smaller
            #    ax.fill_between(t, 0, layered_y[:,0], rasterized=True, linewidth=0)
            #else:
            #    ax.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
            #                    facecolor=color_wheel.get_color(), rasterized=True, \
            #                    linewidth=0)

    if sc.plot_type in ('ss'):
        (xv, yv), z = plot_data
        plt.pcolormesh(xv, yv, z, vmin=0, vmax=1, cmap='prism')
        if sc.solve_type == 'ss_u_transplant':
            plt.xlabel('antibiotic strength')
            plt.ylabel('inoculated C. diff')
        if sc.solve_type == 'ss_duration_dose':
            plt.xlabel('antibiotic strength')
            plt.ylabel('transplant strength')
        plt.gca().tick_params(axis='x', pad=10)
        plt.axis('tight')
        #plt.axis('square')
        #plt.colorbar()
        plt.tight_layout()

    if sc.plot_type == ('3_bw_ss'):
        ((xv1, yv1), z1), ((xv2, yv2), z2), ((xv3, yv3), z3) = plot_data
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        for (index, ((xv, yv), z)) in enumerate([((xv1, yv1), z1), ((xv2, yv2), z2), ((xv3, yv3), z3)]):
            diff1 = np.zeros(np.shape(z))
            for i1, row in enumerate(z):
                current = row[0]
                for i2, item in enumerate(row):
                    if abs(item - current) > .001:
                        diff1[i1][i2] = 1
                    current = item
            diff2 = np.zeros(np.shape(z))
            for i1, column in enumerate(z.T):
                current = column[0]
                for i2, item in enumerate(column):
                    if abs(item - current) > .001:
                        diff2[i2][i1] = 1
                    current = item

            #plt.figure(figsize=(12,5))
            if index == 0: ax = ax1
            elif index == 1: ax = ax2
            elif index == 2: ax = ax3
            ax.axis([0, 2, 0, 1])

            y_ticks = [.25, .5, .75]
            y_labels = ['without\nC. diff', '', 'with\nC. diff']
            if index == 2:
                ax.set_xlabel('antibiotic strength')
                ax.set_xticks(np.arange(0,2.5,.5))
                ax.set_xticklabels(np.arange(0,2.5,.5))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.tick_params(axis='y', which='major', length=0)
            tick_labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_labels = ['']*len(tick_labels)
            if index == 0 or index == 1:
                ax.tick_params(axis='x', which='major', length=0)
                ax.set_xticklabels(empty_labels)

            if index == 0:
                ax.text(.8, .75, 'infected A', ha='center')
                ax.text(.8, .25, 'uninfected B', ha='center')
                ax.text(1.9, .925, 'ICs: 1, 3, 4\n 6, 7, 9 \ ', ha='right', va='top', bbox=dict(boxstyle='round', ec='black', fc='white'))
                ax.text(0, 1.08, '\\textbf{(a) CD-susceptible}', ha='left')
            if index == 1:
                ax.text(.8, .5, 'uninfected C', ha='center', va='center')
                ax.text(1.715, .925, 'ICs: 2, 8', ha='center', va='top', bbox=dict(boxstyle='round', ec='black', fc='white'))
                ax.text(0, 1.08, '\\textbf{(b) CD-resilient}', ha='left')
            if index == 2:
                #ax.text(.3, .5, 'uninfected C', rotation=90, va='center')
                ax.text(.4, .5, 'uninfected C', ha='center', va='center')
                ax.text(.9, .75, 'infected D')
                ax.text(.9, .25, 'uninfected E')
                ax.text(1.715, .925, 'IC: 5', ha='center',va='top', bbox=dict(boxstyle='round', ec='black', fc='white'))
                ax.text(0, 1.08, '\\textbf{(c) CD-fragile}', ha='left')

            xvals = []
            yvals = []
            for i, row in enumerate(diff1):
                for j, item in enumerate(row):
                    if item == 1:
                        assert diff1[i,j] == item
                        try:
                            ax.plot([xv[0][j], xv[0][j]], [yv[i][0], yv[i+1][0]], color='k')
                            xvals.append((xv[0][j], yv[i][0]))
                            xvals.append((xv[0][j], yv[i+1][0]))
                        except IndexError:
                            print("out of bounds")
            for i, row in enumerate(diff2):
                for j, item in enumerate(row):
                    if item == 1:
                        assert diff2[i,j] == item
                        try:
                            ax.plot([xv[0][j], xv[0][j+1]], [yv[i][0], yv[i][0]],
                                    color='k',solid_joinstyle='miter',
                                    solid_capstyle='butt')
                            yvals.append((xv[0][j+1], yv[i][0]))
                            print('plotted horizontal')
                        except IndexError:
                            print("out of bounds")
            clean_xvals = []
            temp_val = -1
            plt.tight_layout(pad=1.2)
            #first_val= xvals[0]

    if sc.plot_type == ('ss_bw'):
        (xv, yv), z = plot_data
        diff1 = np.zeros(np.shape(z))
        for i1, row in enumerate(z):
            current = row[0]
            for i2, item in enumerate(row):
                if abs(item - current) > .001:
                    diff1[i1][i2] = 1
                current = item
        diff2 = np.zeros(np.shape(z))
        for i1, column in enumerate(z.T):
            current = column[0]
            for i2, item in enumerate(column):
                if abs(item - current) > .001:
                    diff2[i2][i1] = 1
                current = item

        #plt.figure(figsize=(12,5))
        if 'fig_2c' in sc.filename:
            fig, ax = plt.subplots(figsize=(10,3.35))
        elif 'fig_2' in sc.filename:
            fig, ax = plt.subplots(figsize=(10,3))
        plt.axis([0, 2, 0, 1])

        if 'fig_2' in sc.filename:
            y_ticks = [.25, .5, .75]
            y_labels = ['without\nC. diff', '', 'with\nC. diff']
            plt.yticks(y_ticks, y_labels)
            plt.tick_params(axis='y', which='major', length=0)
        if 'fig_2c' in sc.filename:
            print('a')
            #plt.xlabel('u')
        elif 'fig_2' in sc.filename:
            tick_labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_labels = ['']*len(tick_labels)
            ax.set_xticklabels(empty_labels)

        if 'fig_2a' in sc.filename:
            plt.text(.8, .75, 'infected A', ha='center')
            plt.text(.8, .25, 'uninfected B', ha='center')
            plt.text(1.9, .9, 'ICs: 1, 3, 4\n 6, 7, 9 \ ', ha='right', va='top', bbox=dict(boxstyle='round', ec='black', fc='white'))
        if 'fig_2b' in sc.filename:
            plt.text(.8, .5, 'uninfected C', ha='center')
            plt.text(1.5, .9, 'ICs: 2, 8', va='top', bbox=dict(boxstyle='round', ec='black', fc='white'))
        if 'fig_2c' in sc.filename:
            plt.text(.3, .5, 'uninfected C', rotation=90, va='center')
            plt.text(.9, .75, 'infected D')
            plt.text(.9, .25, 'uninfected E')
            plt.text(1.59, .9, 'IC: 5', va='top', bbox=dict(boxstyle='round', ec='black', fc='white'))


        #for tick in plt.gca().yaxis.get_major_ticks():
        #    tick.set_verticalalignment("top")
            #tick.tick1line.set_markersize(3)
            #tick.tick2line.set_markersize(3)
            #tick.label1.set_verticalalignment('center')
        #plt.tick_params(axis='y', which='major', length=1)
        #yticks = plt.axes().yaxis.get_major_ticks()
        #yticks[0].label1.set_visible(False)
        #yticks[2].label1.set_visible(False)
        xvals = []
        yvals = []
        for i, row in enumerate(diff1):
            for j, item in enumerate(row):
                if item == 1:
                    assert diff1[i,j] == item
                    try:
                        plt.plot([xv[0][j], xv[0][j]], [yv[i][0], yv[i+1][0]], color='k')
                        xvals.append((xv[0][j], yv[i][0]))
                        xvals.append((xv[0][j], yv[i+1][0]))
                    except IndexError:
                        print("out of bounds")
        for i, row in enumerate(diff2):
            for j, item in enumerate(row):
                if item == 1:
                    assert diff2[i,j] == item
                    try:
                        plt.plot([xv[0][j], xv[0][j+1]], [yv[i][0], yv[i][0]],
                                color='k',solid_joinstyle='miter',
                                solid_capstyle='butt')
                        yvals.append((xv[0][j+1], yv[i][0]))
                        print('plotted horizontal')
                    except IndexError:
                        print("out of bounds")
        clean_xvals = []
        temp_val = -1
        #first_val= xvals[0]

        if ('fig_3' in sc.filename) or ('fig_8' in sc.filename):
            last_val = xvals[-1]
            #xvals.reverse()
            for item in xvals:
                if temp_val != item[0]:
                    clean_xvals.append(item)
                temp_val = item[0]
            #if clean_xvals[-1] != first_val:
            #    clean_xvals.append(first_val)
            if ((clean_xvals[-1][1] < 1) and (last_val[1] == 1)):
                clean_xvals[-1] = last_val
            elif ((clean_xvals[-1][0] < 2) and (clean_xvals[-1][1] < 1)):
                #clean_xvals.insert(0, yvals[-1])
                clean_xvals.append(yvals[-1])
            plt.gca().tick_params(axis='x', pad=10)
            print(clean_xvals)
            return zip(*clean_xvals)
            #plt.tight_layout()

    if sc.plot_type == 'composition_snapshot':
        fig, (ax1, ax2) = plt.subplots(2)#, 1, figsize=(10,3.75))
        for comp_type in ('init','final'):
            print('comp_type', comp_type)
            vals = np.zeros((9, 11))
            for i in range(9):
                scn = Scenario('default'       , i            ,
                               sc.transplant_type_1 , sc.transplant_time_1 , sc.transplant_strength_1,
                               sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                               sc.treatment_type    , sc.treatment_dose    , sc.treatment_duration    ,
                               'microbe_trajectory' , sc.t_end             , False                    ,
                               'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master(scn)
                if comp_type == 'init':
                    vals[i] = y[0]
                elif comp_type == 'final':
                    vals[i] = y[-1]
            total = np.array([sum(column) for column in vals])
            log_total = 11+np.log10(total)
            log_species = np.array( [[log_total[i]*(vals[i][j] / total[i]) for j in
                                      range(11)] for i in range(9)] )
            if comp_type == 'init':
                ax = ax1
            elif comp_type == 'final':
                ax = ax2
            running_tot = np.zeros(9)
            c_diff_index = labels.index("Clostridium difficile")

            import seaborn.apionly as sns
            part_1 = sns.color_palette("Set1", n_colors = 8)
            part_2 = sns.color_palette("Set2", n_colors = 8)
            #red = sns.color_palette("red")[5]
            palette = (part_1[1:] + part_2)
            palette.insert(10, 'red')

            palette_iter = iter(palette)
            #width=.88
            width=.8
            for (m_index, microbe) in enumerate(range(11)):
                ax.bar( np.arange(9), [log_species[j][m_index] for j in range(9)],
                                      bottom=[running_tot[j] for j in range(9)],
                                      width=width,
                                      color=next(palette_iter) )
                running_tot = [running_tot[k] + log_species[k][m_index] for (k, val) in enumerate(running_tot)]
            ax.set_xticklabels( [str(i+1) for i in range(9)] )
            ax.set_yticklabels( ['$10^{%d}$'%i for i in range(0,14,4)] )
            ax.set_xticks(np.arange(9))
            ax.set_yticks([i for i in range(0,14,4)])
            ax.set_ylabel('microbe count')
            ax.axis([-width/2, 8 + width/2, 0, 13])
            if comp_type == 'init':
                ax.set_xlabel('initial state')
                ax.text(-1.9, 14, '\\textbf{(a) Experimentally measured initial conditions} ', ha='left')
            elif comp_type == 'final':
                ax.set_xlabel('steady state')
                ax.text(-1.9, 14, '\\textbf{(b) State after simulated treatment}', ha='left')
            #ax.set_xlabel('initial condition')
            #plt.tick_params(axis='y', which='both', left='off', right='off',
            #        labelleft='off')
            #plt.tick_params(axis='x', which='both', length=0)
            plt.tight_layout()

    if sc.plot_type == 'steady_state_snapshot':
        vals = np.zeros((5, 11))
        for i,x in enumerate(['A', 'B', 'C', 'D', 'E']):
            if x == 'A':
                scn = Scenario('default'       , 0            ,
                               'cdiff'              , 10                   , 1,
                               sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                               sc.treatment_type    , 0                    , sc.treatment_duration    ,
                               'microbe_trajectory' , sc.t_end             , False                    ,
                               'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master(scn)
                vals[i] = y[-1]
            elif x == 'B':
                scn = Scenario('default'       , 0            ,
                               'cdiff'              , 10                   , 0,
                               sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                               sc.treatment_type    , 0                    , sc.treatment_duration    ,
                               'microbe_trajectory' , sc.t_end             , False                    ,
                               'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master(scn)
                vals[i] = y[-1]
            elif x == 'C':
                scn = Scenario('default'       ,  1           ,
                               'cdiff'              , 10                   , 1,
                               sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                               sc.treatment_type    , 0                    , sc.treatment_duration    ,
                               'microbe_trajectory' , sc.t_end             , False                    ,
                               'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master(scn)
                vals[i] = y[-1]
            elif x == 'D':
                scn = Scenario('default'       , 4            ,
                               'cdiff'              , 10                   , 1,
                               sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                               sc.treatment_type    , 1                    , sc.treatment_duration    ,
                               'microbe_trajectory' , sc.t_end             , False                    ,
                               'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master(scn)
                vals[i] = y[-1]
            elif x == 'E':
                scn = Scenario('default'       , 4            ,
                               'cdiff'              , 10                   , 0,
                               sc.transplant_type_2 , sc.transplant_time_2 , sc.transplant_strength_2 ,
                               sc.treatment_type    , 1                    , sc.treatment_duration    ,
                               'microbe_trajectory' , sc.t_end             , False                    ,
                               'none'               , 'basic_traj_log'     , 'new_figs/throwaway.pdf')
                t, y = master(scn)
                vals[i] = y[-1]

        total = np.array([sum(column) for column in vals])
        log_total = 11+np.log10(total)
        log_species = np.array( [[log_total[i]*(vals[i][j] / total[i]) for j in
                                  range(11)] for i in range(5)] )
        fig, ax = plt.subplots(figsize=(10,3.75))
        running_tot = np.zeros(5)
        c_diff_index = labels.index("Clostridium difficile")

        import seaborn.apionly as sns
        part_1 = sns.color_palette("Set1", n_colors = 8)
        part_2 = sns.color_palette("Set2", n_colors = 8)
        #red = sns.color_palette("red")[5]
        palette = (part_1[1:] + part_2)
        palette.insert(10, 'red')

        palette_iter = iter(palette)
        #width=.88
        width=.8
        for (m_index, microbe) in enumerate(range(11)):
            ax.bar( np.arange(5), [log_species[j][m_index] for j in range(5)],
                                  bottom=[running_tot[j] for j in range(5)],
                                  width=width,
                                  color=next(palette_iter) )
            running_tot = [running_tot[k] + log_species[k][m_index] for (k, val) in enumerate(running_tot)]
        ax.set_xticklabels( ['A', 'B', 'C', 'D', 'E'] )
        ax.set_yticklabels( ['$10^{%d}$'%i for i in range(0,14,4)] )
        ax.set_xticks(np.arange(5))
        ax.set_yticks([i for i in range(0,14,4)])
        ax.set_ylabel('microbe count')
        ax.axis([-width/2, 4 + width/2, 0, 13])
        ax.set_xlabel('reachable steady states')
        #ax.set_xlabel('initial condition')
        #plt.tick_params(axis='y', which='both', left='off', right='off',
        #        labelleft='off')
        plt.tick_params(axis='x', which='both', length=0)
        plt.tight_layout()

    if sc.if_plot == 'save':
        plt.savefig(sc.filename, dpi=300)
        print('SAVED', sc.plot_type, 'plot for', sc.scenario_type, 'scenario to', sc.filename)
    if sc.if_plot == 'show':
        plt.show()
        print('SHOWING', sc.plot_type, 'plot for scenario', sc.scenario_type)


# various plots for standard model
def plot_standard(plot_vars, plot_type, filename):
    import matplotlib.pyplot as plt
    plt.clf()
    if (plot_type == 'mutant_new_order'):
        t, y, labels, (aaa, if_plot) = plot_vars
        plt.plot(t, y[:,8], 'b')
        plt.plot(t, y[:,8]+y[:,-1], 'k')
        plt.xlabel('$t$ (days)')
        plt.ylabel('population')
        x1,x2,y1,y2 = plt.axis()
        ax = plt.gca()
        ax.set_ylim([0, y2])
        ax.set_xlim([0, x2])
    if (plot_type == 'basic_biodiv'):
        t, y, labels, if_plot = plot_vars
        sns.set_palette(sns.color_palette("Set1", n_colors=len(labels)+4))

        y_total = np.array([sum(y[i]) for i in range(len(t))])
        biodiv = [-sum([(y[i,j]/y_total[i])*math.log10(y[i,j]/y_total[i]) if \
                  (abs(y[i,j]) > 0.00001) else 0 for j in range(len(y[0]))]) for i in range(len(t))]
        plt.plot(t, biodiv, 'b')
        plt.xlabel('$t$ (days)')
        plt.ylabel('Shannon biodiversity')
        x1,x2,y1,y2 = plt.axis()
        ax = plt.gca()
        ax.set_ylim([0, y2])

    if (plot_type == 'sporulation'):
        t, y, labels, (if_legend, if_plot) = plot_vars
        plt.plot(t, y[:,-1])
    if (plot_type == 'plot_u'):
        (t, y, [dose, duration, u_type], if_plot) = plot_vars
        plt.plot(t, u(t, dose, duration, u_type))
        plt.xlabel('$t$ (days)')
        plt.ylabel('$u(t)$')
        plt.tight_layout()
        plt.axis([t[0], t[-1], 0, 1])
    if (plot_type == 'basic_traj_log'):
        t, y, labels, (if_legend, if_plot) = plot_vars
        #sns.set_palette(sns.color_palette("Set2", n_colors=20))
        y_total = np.array([sum(y[i]) for i in range(len(t))])
        y_log_total = 11 + np.log10(y_total) # +11 to compensate for microbe units
        y_log_species = np.array([(y[i,:]/y_total[i])*y_log_total[i] for i in \
                range(len(t))])
        layered_y = np.zeros(np.shape(y))
        layered_y[:,0] = y_log_species[:,0]
        for i in range(1,len(labels)):
            layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]
        if if_plot == 'multi':
            fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
        else:
            fig, ax = plt.subplots()
        c_diff_index = labels.index("Clostridium difficile")
        for i in range(0,len(labels)):
            if i==c_diff_index:
                color_wheel, = ax.plot(t, layered_y[:,i], linewidth=2, color='r')
            elif i==(3+c_diff_index):
                color_wheel, = ax.plot(t, layered_y[:,i], linewidth=2, color='k')
            else:
                color_wheel, = ax.plot(t, layered_y[:,i], linewidth=2)
            if i == 0:
                ax.fill_between(t, 0, layered_y[:,0], rasterized=True, \
                        linewidth=0)
            else:
                ax.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
                        facecolor=color_wheel.get_color(), rasterized=True,\
                        linewidth=0)
        ax.set_xlabel('$t$ (days)')
        ax.set_ylabel('log$_{10}$ microbe concentration')
        ax.set_xlim([0, t[-1]])
        ax.set_ylim([0, 1.05*max(y_log_total)])
        #plt.tight_layout()

        if if_legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height*0.1, box.width,
                box.height*0.9])
            plt.legend(labels, prop={'size':6}, loc='upper center', ncol=3,
                    bbox_to_anchor=(0.45, -0.15), fancybox=True)
    if (plot_type == 'basic_traj'):
        t, y, labels, if_plot = plot_vars

        sns.set_palette(sns.color_palette("Set2", len(labels)+1))
        layered_y = np.zeros(np.shape(y))
        layered_y[:,0] = y[:,0]
        for i in range(1,len(labels)):
            layered_y[:, i] = layered_y[:,i-1] + y[:,i]
        fig, ax = plt.subplots()
        for i in range(0,len(labels)):
            color_wheel, = plt.plot(t, layered_y[:,i], linewidth=2)
            if i == 0:
                ax.fill_between(t, 0, layered_y[:,0], rasterized=True, \
                        linewidth=0)
            else:
                ax.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
                        facecolor=color_wheel.get_color(), rasterized=True,\
                        linewidth=0)
        plt.legend(labels, prop={'size':7}, loc=9, ncol=3)
        plt.xlabel('$t$ (days)')
        plt.tight_layout()
    if (plot_type == 'ss_basins'):
        (xv, yv), z, labels, (if_legend, if_plot) = plot_vars
        #a, z, labels, if_plot = plot_vars
        #print('aaa',a)
        #fig, ax = plt.subplots()
        plt.pcolormesh(xv, yv, z, vmin=0, vmax=1, cmap='prism')
        #x_fit_i = np.linspace(0.001,.5,100)
        #x_fit = [x_fit_i[i] for i in range(len(x_fit_i)) if (.719/x_fit_i[i]<8)]
        #y_fit = [.719/x_fit[i] for i in range(len(x_fit))]
        #ax.plot(x_fit, y_fit,'white')
        if if_plot != 'none':
            plt.xlabel('antibiotic density')
            plt.ylabel('antibiotic duration')
        plt.axis('tight')
        plt.colorbar()
        plt.tight_layout()
    if (plot_type == 'just_ics'):
        ic, labels, if_plot = plot_vars
        plt.axis('off')
        t = np.array([0, 1])
        y = np.array([ic, ic])
        y_total = np.array([sum(y[i]) for i in range(len(t))])
        y_log_total = 11 + np.log10(y_total) # +11 to compensate for microbe units
        y_log_species = np.array([(y[i,:]/y_total[i])*y_log_total[i] for i in \
                range(len(t))])
        layered_y = np.zeros(np.shape(y))
        layered_y[:,0] = y_log_species[:,0]
        for i in range(1,len(labels)):
            layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]
        if if_plot == 'none':
            #fig, ((a, ax, b), (c,d,e),(f,g,h)) = plt.subplots(nrows=3, ncols=3)
            print('here aa')
        #else:
            #fig, ax = plt.subplots()
        c_diff_index = labels.index("Clostridium difficile")
        for i in range(0,len(labels)):
            if i==c_diff_index:
                color_wheel, = plt.plot(t, layered_y[:,i], linewidth=0, color='r')
            elif i==(c_diff_index+3):
                color_wheel, = plt.plot(t, layered_y[:,i], linewidth=0, color='k')
            else:
                color_wheel, = plt.plot(t, layered_y[:,i], linewidth=0)
            if i == 0:
                plt.fill_between(t, 0, layered_y[:,0], rasterized=True, \
                        linewidth=0)
            else:
                plt.fill_between(t, layered_y[:,i-1], layered_y[:,i], \
                        facecolor=color_wheel.get_color(), rasterized=True,\
                        linewidth=0)
        plt.axis([0, 1, 0, layered_y.max()*1.1])
        #ax.set_xlabel('$t$ (days)')
        #ax.set_ylabel('log$_{10}$ microbe concentration')
        plt.tight_layout()
    if (plot_type == 'just_legend'):
        t, y, labels, (if_legend, if_plot) = plot_vars
        ax = plt.subplot()
        y_total = np.array([sum(y[i]) for i in range(len(t))])
        y_log_total = 11 + np.log10(y_total) # +11 to compensate for microbe units
        y_log_species = np.array([(y[i,:]/y_total[i])*y_log_total[i] for i in \
                range(len(t))])
        layered_y = np.zeros(np.shape(y))
        layered_y[:,0] = y_log_species[:,0]
        for i in range(1,len(labels)):
            layered_y[:, i] = layered_y[:,i-1] + y_log_species[:,i]
        c_diff_index = labels.index("Clostridium difficile")
        for i in range(0,len(labels)):
            if i==c_diff_index:
                color_wheel, = ax.plot([0], [0], linewidth=2, color='r')
            elif i==11:
                color_wheel, = ax.plot([0], [0], linewidth=2, color='k')
            else:
                color_wheel, = ax.plot([0], [0], linewidth=2)
        ax.set_axis_off()
        ax.legend(labels, prop={'size':8}, loc='center', ncol=2,
                fancybox=True)
    if (if_plot == 'show'):
        plt.show()
        print('showing', plot_type)
    if (if_plot == 'save'):
        plt.savefig(filename)
        print('    saved', plot_type, 'to', filename, '    ')

# premade plots
def example_plots(example_type):
    ###if example_type == 'default':
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)
    ###    ic_num = 4 # initial conditions 0-8 available
    ###    ic_default = parse_ic((ic_data, ic_num), 'default')

    ###    # setup and solve ODE
    ###    dose=0.1; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=300;
    ###    params_default = (mu_default, M_default, eps_default, u_params)
    ###    solve_vars = (ic_default, *params_default, t_end)
    ###    t, y = solve(solve_vars, solve_type='default')

    ###    print(np.sqrt(np.dot(y[-1],y[-1])))

    ###    ## plot
    ###    #if_plot = 'save' # 'save' or 'show'
    ###    #plot_vars = (t, y, labels, (True, if_plot))
    ###    #plot_standard(plot_vars, 'basic_traj_log', 'new_figs/example_default_a.pdf')
    ###if example_type == 'basic_mutation':
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data,
    ###            model_type='mutation')
    ###    ic_num = 0 # initial conditions 0-8 available
    ###    ic_default = parse_ic((ic_data, ic_num), 'mutation')

    ###    for ii in range(9):
    ###        ic_default = parse_ic((ic_data, ii), 'mutation')
    ###        print('ic', ii, 'has norm', sum(ic_default))

    ###    ic_default[8] = 5
    ###    cc = 2.268078
    ###    multiplier = .99
    ###    mu_default[-1] = mu_default[-1]*multiplier
    ###    M_default[-1,:] = M_default[-1,:]*multiplier

    ###    xm = (cc*(M_default[-1,8] - M_default[8,8])+mu_default[-1]-mu_default[8])/(M_default[-1,8]-M_default[-1,-1])
    ###    xc = cc-xm
    ###    print('xm is', xm, 'xc is', xc)
    ###            # setup and solve ODE
    ###    dose=3.0; duration=30.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=300;
    ###    params_default = (mu_default, M_default, eps_default, u_params)
    ###    solve_vars = (ic_default, *params_default, t_end)
    ###    t, y = solve(solve_vars, solve_type='mutation')
    ###    print('c diff is', y[-1,8])

    ###    # params for changing 3 M values
    ###    ##eps_default[8] = 2*eps_default[8]
    ###    #ic_default[8] = 5
    ###    #cc = 2.268078
    ###    #multiplier = .95
    ###    #diff = mu_default[-1]*(1-multiplier)
    ###    #mu_default[-1] = mu_default[-1]*multiplier
    ###    #print('Mcc is', M_default[8,8])
    ###    #print('Mcm is', M_default[8,8])
    ###    #M_default[-1,8] = -.02
    ###    #M_default[-1,-1] = -.06
    ###    #xm = (cc*(M_default[-1,8] - M_default[8,8])+mu_default[-1]-mu_default[8])/(M_default[-1,8]-M_default[-1,-1])
    ###    #xc = cc-xm
    ###    #print('xm is', xm, 'xc is', xc)
    ###    ##M_default[-1,8] = M_default[-1,8]*multiplier
    ###    ##M_default[-1,-1] = M_default[-1,-1]*multiplier
    ###    #print('diff is', diff)

    ###    ## setup and solve ODE
    ###    #dose=3.0; duration=30.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    #t_end=300;
    ###    #params_default = (mu_default, M_default, eps_default, u_params)
    ###    #solve_vars = (ic_default, *params_default, t_end)
    ###    #t, y = solve(solve_vars, solve_type='mutation')
    ###    #print('c diff is', y[-1,8])

    ###    # plot
    ###    if_plot = 'save' # 'save' or 'show'
    ###    plot_vars = (t, y, labels, (True, if_plot))
    ###    plot_standard(plot_vars, 'basic_traj_log', 'new_figs/mutant_m_by_constant.pdf')
    ###    plot_standard(plot_vars, 'mutant_new_order', 'new_figs/mutant_m_by_constant_zoom.pdf')
    ###if example_type == 'conc_vs_duration':
    ###    ### PLOT FOR FIG 3
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)
    ###    ic_num = 4 # initial conditions 0-8 available
    ###    ic_default = parse_ic((ic_data, ic_num), 'default')

    ###    for uu in ('default', 'tapered'):#, 'pulse'):
    ###        # setup and solve ODE
    ###        dose=2.0; duration=1.0; u_type=uu; u_params=[dose, duration, u_type]
    ###        t_end=1000;
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        transplant_time=0.001; strength=1; sample_of=0
    ###        sample_vars = (transplant_time, strength, labels, sample_of)

    ###        solve_vars = (ic_default, *params_default, t_end, sample_vars,
    ###                transplant_time)
    ###        if uu=='default':
    ###            t1, y1 = solve(solve_vars, solve_type='ss_basins')
    ###        if uu=='tapered':
    ###            t2, y2 = solve(solve_vars, solve_type='ss_basins')

    ###    # plot
    ###    if_plot = 'save' # 'save' or 'show'
    ###    plot_vars = (t1, np.abs(y1-y2), labels, (False, if_plot))
    ###    plot_standard(plot_vars, 'ss_basins',
    ###            'new_figs/u_dens_tradeoff_diff_def_tap_2.pdf')
    ###if example_type == 'conc_vs_duration_multiplot':
    ###    ### PLOT FOR FIG 2
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)
    ###    ic_num = 4 # initial conditions 0-8 available
    ###    ic_default = parse_ic((ic_data, ic_num), 'default')

    ###    # setup and solve ODE
    ###    dose=1.0; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=6000;
    ###    params_default = (mu_default, M_default, eps_default, u_params)
    ###    transplant_time=10; strength=1; sample_of=0
    ###    sample_vars = (transplant_time, strength, labels, sample_of)

    ###    solve_vars = [ic_default, *params_default, t_end, sample_vars,
    ###            transplant_time]
    ###    solve_type='ss_basins_u_cdiff'
    ###    #t, y = solve(solve_vars, solve_type='ss_basins_u_cdiff')

    ###    # plot
    ###    if_plot = 'none' # 'save' or 'show'
    ###    #plot_vars = (t, y, labels, (False, if_plot))
    ###    plot_vars = (labels, (False, if_plot))
    ###    plot_type='ss_basins'
    ###    #plot_standard(plot_vars, 'ss_basins', 'new_figs/u_dens_test.pdf')
    ###    filename = 'new_figs/multiplot_u_cdiff_prism.pdf'
    ###    multiplot((solve_vars, plot_vars), (solve_type, plot_type), filename, 'solve_all_ics')
    ###if example_type == 'conc_vs_duration_multiplot_normalized':
    ###    ### PLOT FOR NORMALIZED PHASE DIAGRAMS
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)
    ###    ic_num = 4 # initial conditions 0-8 available
    ###    ic_default = parse_ic((ic_data, ic_num), 'default')

    ###    # setup and solve ODE
    ###    dose=1.0; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=6000;
    ###    params_default = (mu_default, M_default, eps_default, u_params)
    ###    transplant_time=10; strength=1; sample_of=0
    ###    sample_vars = (transplant_time, strength, labels, sample_of)

    ###    solve_vars = [ic_default, *params_default, t_end, sample_vars,
    ###            transplant_time]
    ###    solve_type='ss_basins_u_cdiff'
    ###    #t, y = solve(solve_vars, solve_type='ss_basins_u_cdiff')

    ###    # plot
    ###    if_plot = 'none' # 'save' or 'show'
    ###    #plot_vars = (t, y, labels, (False, if_plot))
    ###    plot_vars = (labels, (False, if_plot))
    ###    plot_type='ss_basins'
    ###    #plot_standard(plot_vars, 'ss_basins', 'new_figs/u_dens_test.pdf')
    ###    filename = 'new_figs/multiplot_normal_u_cdiff_prism_4.pdf'
    ###    multiplot((solve_vars, plot_vars), (solve_type, plot_type), filename, 'solve_all_ics_normal')
    ###if example_type == 'plot_u':
    ###    import matplotlib.pyplot as plt
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)
    ###    ic_num = 2 # initial conditions 0-8 available
    ###    ic_default = parse_ic((ic_data, ic_num), 'default')

    ###    for uu in ('default', 'tapered', 'pulse'):
    ###        # setup and solve ODE
    ###        dose=3.0; duration=15.0; u_type=uu; u_params=[dose, duration, u_type]
    ###        t_end=30;
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        solve_vars = (ic_default, *params_default, t_end)
    ###        t, y = solve(solve_vars, solve_type='default')

    ###        # plot
    ###        if_plot = 'save' # 'save' or 'show'
    ###        plot_vars = (t, y, u_params, if_plot)
    ###        plot_standard(plot_vars, 'plot_u', 'new_figs/u_type_'+uu+'.pdf')
    ###if example_type == 'show_vulnerable':
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)

    ###    different_ss = {}
    ###    different_size = {}
    ###    for ss in ('3_sus', '4_invul', '4_sus'):
    ###        if ss == '3_sus':
    ###            ic_num = 3
    ###            dose = 0.0
    ###        if ss == '4_sus':
    ###            ic_num = 4
    ###            dose = 1.0
    ###        if ss == '4_invul':
    ###            ic_num = 4
    ###            dose = 0.0
    ###        #ic_num = 2 # initial conditions 0-8 available
    ###        ic_default = parse_ic((ic_data, ic_num), 'default')

    ###        # setup and solve ODE
    ###        #dose=0.0; 
    ###        duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###        t_end=500;
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        solve_vars = (ic_default, *params_default, t_end)
    ###        t, y = solve(solve_vars, solve_type='default')

    ###        # plot
    ###        if_plot = 'save' # 'save' or 'show'
    ###        plot_vars = (t, y, labels, (True, if_plot))
    ###        plot_standard(plot_vars, 'basic_traj_log', 'new_figs/example_default.pdf')
    ###        print('ss is', ss)
    ###        print('y[-1] is', y[-1])
    ###        print('M*y is', np.dot(M_default, y[-1]))
    ###        temp_val = np.dot(M_default, y[-1]) + mu_default
    ###        print('(M+mu)*y is', np.dot(M_default, y[-1]) + mu_default)
    ###        print()
    ###        different_ss[ss] = temp_val[8]
    ###        different_size[ss] = np.sum(y[-1])
    ###    print(different_ss)
    ###    print(different_size)
    ###if example_type == 'misc_transplants':
    ###    #example_plots('default')
    ###    import matplotlib.pyplot as plt
    ###    ics=4
    ###    tic = time.clock()

    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data)
    ###    ic_type = 'default'; ic_num = ics # initial conditions 0-8 available 
    ###                                    # ic 4 gives c. diff characteristics!
    ###    ic_vars = (ic_data, ic_num)
    ###    ic_default = parse_ic(ic_vars, ic_type)

    ###    if timing: print("time elapsed for parameters is", time.clock()-tic)
    ###    tic = time.clock()

    ###    # setup and solve ODE
    ###    dose=0.7; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=1000; transplant_time=10; strength=1; sample_of=0
    ###    for sample_of in (7,):
    ###        print('transplant of type', sample_of)
    ###        sample_vars = (transplant_time, strength, labels, sample_of)
    ###        #sample = generate_sample(sample_vars, sample_of)
    ###        sample = generate_sample(sample_vars, 'ic_plus_cdiff')
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        solve_vars = [ic_default, *params_default, t_end, sample_vars, transplant_time] # for solve_type 'ss_basins'
    ###        #solve_vars = [ic_default, *params_default, t_end, sample, transplant_time] # for solve_type 'transplant'
    ###        solve_type='ss_basins_delay'
    ###        t, y = solve(solve_vars, solve_type)
    ###        #print('solving ic number', ics)
    ###        #(xx, yy), z = solve(solve_vars, solve_type)
    ###        #print(z)
    ###        if timing: print("time elapsed for solving is", time.clock()-tic)
    ###        tic = time.clock()



    ###        # plot
    ###        if_legend = False
    ###        #if_plot = 'none' # 'save' or 'show' or 'none'
    ###        if_plot = 'save' # 'save' or 'show' or 'none'
    ###        #plot_vars_end = (labels, if_plot)
    ###        plot_vars = (t, y, labels, (if_legend, if_plot))
    ###        plot_vars_end = (labels, if_plot)
    ###        plot_type='ss_basins'
    ###        filename='new_figs/new_transplant_compare.pdf'
    ###        #plot_type='basic_traj_log'; filename='new_figs/layered_default_2.pdf'
    ###        plot_standard(plot_vars, plot_type, filename)
    ###        #multiplot((solve_vars, plot_vars_end), (solve_type, plot_type), filename, 'solve_all_ics')
    ###    #save_metadata(filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars)
    ###    #plot_type='basic_biodiv'; filename='new_figs/layered_default_biodiv.pdf'
    ###    #plot_standard(plot_vars, plot_type, filename)
    ###    #save_metadata(filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars)

    ###    #plot_varss = (plot_vars, plot_vars); plot_types=('basic_traj_log','basic_biodiv')
    ###    #multiplot(plot_varss, plot_types, 'new_figs/ic4_clindamycin_p7_cdiff_1.pdf')

    ###    # example load of metadata
    ###    #filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars \
    ###    #            = load_metadata('new_figs/.layered_default.pdf')
    ###    #plt.savefig('new_figs/ic_4_u_varies_phase_2.pdf')
    ###    #print('    saved', plot_type, 'to', 'new_figs/all_u_density_duration_long_term_2.pdf', '    ')

    ###    if timing: print("time elapsed for plotting is", time.clock()-tic)
    ###    print("done in", time.clock()-tic0, "seconds")
    ###if example_type == 'spor_w_transplant':
    ###    #example_plots('default')
    ###    import matplotlib.pyplot as plt
    ###    ics=4
    ###    tic = time.clock()

    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data, \
    ###            model_type='sporulation')
    ###    ic_type = 'sporulation'; ic_num = ics # initial conditions 0-8 available 
    ###                                    # ic 4 gives c. diff characteristics!
    ###    ic_vars = (ic_data, ic_num)
    ###    ic_default = parse_ic(ic_vars, ic_type)

    ###    if timing: print("time elapsed for parameters is", time.clock()-tic)
    ###    tic = time.clock()

    ###    # setup and solve ODE
    ###    dose=0.7; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=1000; transplant_time=10; strength=1; sample_of=0
    ###    for sample_of in (7,):
    ###        print('transplant of type', sample_of)
    ###        sample_vars = (transplant_time, strength, labels, sample_of)
    ###        #sample = generate_sample(sample_vars, sample_of)
    ###        sample = generate_sample(sample_vars, 'ic_plus_cdiff')
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        solve_vars = [ic_default, *params_default, t_end, sample_vars, transplant_time] # for solve_type 'ss_basins'
    ###        #solve_vars = [ic_default, *params_default, t_end, sample, transplant_time] # for solve_type 'transplant'
    ###        solve_type='ss_basins_delay'
    ###        t, y = solve(solve_vars, solve_type)
    ###        #print('solving ic number', ics)
    ###        #(xx, yy), z = solve(solve_vars, solve_type)
    ###        #print(z)
    ###        if timing: print("time elapsed for solving is", time.clock()-tic)
    ###        tic = time.clock()



    ###        # plot
    ###        if_legend = False
    ###        #if_plot = 'none' # 'save' or 'show' or 'none'
    ###        if_plot = 'save' # 'save' or 'show' or 'none'
    ###        #plot_vars_end = (labels, if_plot)
    ###        plot_vars = (t, y, labels, (if_legend, if_plot))
    ###        plot_vars_end = (labels, if_plot)
    ###        plot_type='ss_basins'
    ###        filename='new_figs/spor_w_transplant_default.pdf'
    ###        #plot_type='basic_traj_log'; filename='new_figs/layered_default_2.pdf'
    ###        plot_standard(plot_vars, plot_type, filename)
    ###        #multiplot((solve_vars, plot_vars_end), (solve_type, plot_type), filename, 'solve_all_ics')
    ###    #save_metadata(filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars)
    ###    #plot_type='basic_biodiv'; filename='new_figs/layered_default_biodiv.pdf'
    ###    #plot_standard(plot_vars, plot_type, filename)
    ###    #save_metadata(filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars)

    ###    #plot_varss = (plot_vars, plot_vars); plot_types=('basic_traj_log','basic_biodiv')
    ###    #multiplot(plot_varss, plot_types, 'new_figs/ic4_clindamycin_p7_cdiff_1.pdf')

    ###    # example load of metadata
    ###    #filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars \
    ###    #            = load_metadata('new_figs/.layered_default.pdf')
    ###    #plt.savefig('new_figs/ic_4_u_varies_phase_2.pdf')
    ###    #print('    saved', plot_type, 'to', 'new_figs/all_u_density_duration_long_term_2.pdf', '    ')

    ###    if timing: print("time elapsed for plotting is", time.clock()-tic)
    ###    print("done in", time.clock()-tic0, "seconds")
    ###if example_type == 'targeted_antibiotic_sporulation':
    ###    # build parameters
    ###    var_data, ic_data = import_data()
    ###    labels, mu_default, M_default, eps_default = parse_data(var_data, \
    ###            model_type='sporulation')
    ###    #eps_default[8] = eps_default[8]*5
    ###    ic_num = 0 # initial conditions 0-8 available
    ###    old_ic_default = parse_ic((ic_data, ic_num), 'sporulation')

    ###    transplant_time=10; strength=1.0; sample_of=False
    ###    sample_vars = (transplant_time, strength, labels, sample_of)
    ###    ic_cdiff_supplement = generate_sample(sample_vars, 'c_diff')
    ###    ic_default = old_ic_default + ic_cdiff_supplement

    ###    # setup and solve ODE
    ###    dose=1.0; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    t_end=200;
    ###    params_default = (mu_default, M_default, eps_default, u_params)
    ###    solve_vars = (ic_default, *params_default, t_end)
    ###    t, y = solve(solve_vars, solve_type='sporulation')

    ###    if_plot = 'save' # 'save' or 'show'
    ###    infected_state = y[-1]
    ###    infected_snapshot_vars = infected_state, labels, if_plot
    ###    #plot_standard(infected_snapshot_vars, 'just_ics', 'new_figs/infected_snapshot.pdf')

    ###    transplant_time=0.01; strength=(0.01); sample_of=1
    ###    sample_vars = (transplant_time, strength, labels, sample_of)
    ###    ic_supplement = np.append(generate_sample(sample_vars, 'ic'), 0)

    ###    fake_eps = np.array([2, 0, 2, 0, -2, -2, -2, -2, -4, 0, -2, 0])
    ###    # fake_eps pushes towards resilient s.s. and away from vulnerable
    ###    eps_default = fake_eps

    ###    dose=1.0; duration=10.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    #dose=1.7; duration=10.0; u_type='default'; u_params=[dose, duration, u_type]
    ###    #dose=1.7; duration=10.0; u_type='pulse'; u_params=[dose, duration, u_type]
    ###    #dose=1.7; duration=10.0; u_type='tapered'; u_params=[dose, duration, u_type]
    ###    t_end=50;
    ###    params_default = (mu_default, M_default, eps_default, u_params)
    ###    solve_vars = (infected_state, *params_default, t_end, ic_supplement,
    ###            transplant_time)
    ###    t, y = solve(solve_vars, solve_type='transplant_spor')

    ###    # plot
    ###    if_legend = False
    ###    plot_vars = (t, y, labels, (if_legend, if_plot))
    ###    plot_standard(plot_vars, 'basic_traj_log',
    ###            'new_figs/comparison_no_spor_2.pdf')
    ###    #plot_standard(plot_vars, 'sporulation', 'new_figs/example_default.pdf')

    ###    print('total time elapsed is', time.clock() - tic0)
    if example_type == 'master_example':
        # choose scenario: 'default' or 'sporulation' or 'mutation_1' or 'mutation_2'
        master_scenario = 'default'
        # choose which initial condition to use (0-8 available)
        ic_num = 4

        # import parameters for gLV model
        global labels, mu, M, eps, ic_data
        var_data, ic_data  = import_data()
        labels, mu, M, eps = parse_data(var_data, scenario = master_scenario)
        ic                 = parse_ic((ic_data, ic_num), scenario = master_scenario)

        # choose parameters for first supplement
        # (supplement may be inoculation w/ C. diff, or transplant)
        global transplant_time_1, transplant_time_2, transplant_type_1, \
               transplant_type_2, transplant_strength_1, transplant_strength_2
        transplant_type_1   = 'cdiff' # 0-8 (corresponding to ICs), 'cdiff', or 'none'
        transplant_time_1   = 10; transplant_strength_1 = 1.0;
        transplant_vars_1   = [transplant_type_1, transplant_time_1, transplant_strength_1]

        # choose parameters for second supplement
        transplant_type_2   = 'cdiff' #'none' # 0-8 (corresponding to ICs), 'cdiff', or 'none'
        transplant_time_2   = 20; transplant_strength_2 = 1.0;
        transplant_vars_2   = [transplant_type_2, transplant_time_2, transplant_strength_2]

        # choose antibiotic treatment parameters
        global treatment_params
        treatment_type     = 'constant' # 'constant', 'tapered', or 'pulsed'
        treatment_dose     = 1.0; treatment_duration = 1.0
        treatment_params   = [treatment_type, treatment_dose, treatment_duration]

        # choose parameters for solving gLV equations
        # solve_type can be: 
        #   'microbe_trajectory' (microbe trajectories through time),
        #   'ss_u_transplant' (steady states, x-axis varies antibiotic, y-axis
        #                      amount of transplant),
        #   'ss_duration_dose' (steady states, x-axis varies treatment_duration,
        #                      y-axis treatment_dose)
        solve_type = 'microbe_trajectory'; t_end = 200;
        t, y       = master_solve(ic, t_end, master_scenario, solve_type)

        # plot
        if_legend = True; if_plot='save'
        m_plot((t,y), (if_legend, if_plot), 'basic_traj_log', 'new_figs/m_plot.pdf', master_scenario)
        #plot_standard(plot_vars, 'basic_traj_log',
        #        'new_figs/not_lots_of_c_diff.pdf')

        #if_plot = 'save' # 'save' or 'show'
        #infected_state = y[-1]
        #infected_snapshot_vars = infected_state, labels, if_plot
        ##plot_standard(infected_snapshot_vars, 'just_ics', 'new_figs/infected_snapshot.pdf')

        #transplant_time=0.01; strength=(0.01); sample_of=1
        #sample_vars = (transplant_time, strength, labels, sample_of)
        #ic_supplement = np.append(generate_sample(sample_vars, 'ic'), 0)

        #fake_eps = np.array([2, 0, 2, 0, -2, -2, -2, -2, -4, 0, -2, 0])
        ## fake_eps pushes towards resilient s.s. and away from vulnerable
        #eps_default = fake_eps

        #dose=1.0; duration=10.0; u_type='default'; u_params=[dose, duration, u_type]
        ##dose=1.7; duration=10.0; u_type='default'; u_params=[dose, duration, u_type]
        ##dose=1.7; duration=10.0; u_type='pulse'; u_params=[dose, duration, u_type]
        ##dose=1.7; duration=10.0; u_type='tapered'; u_params=[dose, duration, u_type]
        #t_end=50;
        #params_default = (mu_default, M_default, eps_default, u_params)
        #solve_vars = (infected_state, *params_default, t_end, ic_supplement,
        #        transplant_time)
        #t, y = solve(solve_vars, solve_type='transplant_spor')

        ## plot
        #if_legend = False
        #plot_vars = (t, y, labels, (if_legend, if_plot))
        #plot_standard(plot_vars, 'basic_traj_log',
        #        'new_figs/comparison_no_spor_2.pdf')
        ##plot_standard(plot_vars, 'sporulation', 'new_figs/example_default.pdf')

        #print('total time elapsed is', time.clock() - tic0)
    ###    if example_type == 'ss_from_infected':
    ###        # build parameters
    ###        var_data, ic_data = import_data()
    ###        labels, mu_default, M_default, eps_default = parse_data(var_data, \
    ###                model_type='sporulation')
    ###        #eps_default[8] = eps_default[8]*5
    ###        ic_num = 0 # initial conditions 0-8 available
    ###        old_ic_default = parse_ic((ic_data, ic_num), 'sporulation')
    ###
    ###        transplant_time=10; strength=1.0; sample_of=False
    ###        sample_vars = (transplant_time, strength, labels, sample_of)
    ###        ic_cdiff_supplement = generate_sample(sample_vars, 'c_diff')
    ###        ic_default = old_ic_default + ic_cdiff_supplement
    ###
    ###        # setup and solve ODE
    ###        dose=1.0; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###        t_end=200;
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        solve_vars = (ic_default, *params_default, t_end)
    ###        t, y = solve(solve_vars, solve_type='sporulation')
    ###
    ###        if_plot = 'save' # 'save' or 'show'
    ###        infected_state = y[-1]
    ###        infected_snapshot_vars = infected_state, labels, if_plot
    ###        plot_standard(infected_snapshot_vars, 'just_ics', 'new_figs/infected_snapshot.pdf')
    ###
    ###        transplant_time=0.01; strength=(0.01); sample_of=1
    ###        sample_vars = (transplant_time, strength, labels, sample_of)
    ###        ic_supplement = np.append(generate_sample(sample_vars, 'ic'), 0)
    ###
    ###        #fake_eps = np.array([2, 0, 2, 0, -2, -2, -2, -2, -4, 0, -2, 0])
    ###        # fake_eps pushes towards resilient s.s. and away from vulnerable
    ###        #eps_default = fake_eps
    ###
    ###        #dose=1.7; duration=10.0; u_type='default'; u_params=[dose, duration, u_type]
    ###        #dose=1.7; duration=10.0; u_type='pulse'; u_params=[dose, duration, u_type]
    ###
    ###        for u_type in ('default', 'tapered', 'pulse'):
    ###            dose=1.7; duration=10.0; u_params=[dose, duration, u_type]
    ###            t_end=500;
    ###            params_default = (mu_default, M_default, eps_default, u_params)
    ###            solve_vars = (infected_state, *params_default, t_end, sample_vars,
    ###                    transplant_time)
    ###            #if u_type == 'default':
    ###            #    t1, y1 = solve(solve_vars, solve_type='ss_basins')
    ###            #elif u_type == 'tapered':
    ###            #    t2, y2 = solve(solve_vars, solve_type='ss_basins')
    ###            t, y = solve(solve_vars, solve_type='ss_basins')
    ###            # plot
    ###            if_legend = True
    ###            #plot_vars = (t1, np.abs(y1-y2), labels, (if_legend, if_plot))
    ###            plot_vars = (t, y, labels, (if_legend, if_plot))
    ###            new_filename = 'new_figs/real_antibiotic_ss_' + u_type  + '_no_spor.pdf'
    ###            plot_standard(plot_vars, 'ss_basins', new_filename)
    ###            #plot_standard(plot_vars, 'sporulation', 'new_figs/example_default.pdf')
    ###
    ###            print('total time elapsed is', time.clock() - tic0)
    ###    if example_type == 'make_legend':
    ###        # build parameters
    ###        var_data, ic_data = import_data()
    ###        labels, mu_default, M_default, eps_default = parse_data(var_data, \
    ###                model_type='sporulation')
    ###        #eps_default[8] = eps_default[8]*5
    ###        ic_num = 0 # initial conditions 0-8 available
    ###        old_ic_default = parse_ic((ic_data, ic_num), 'sporulation')
    ###
    ###        transplant_time=10; strength=1.0; sample_of=False
    ###        sample_vars = (transplant_time, strength, labels, sample_of)
    ###        ic_cdiff_supplement = generate_sample(sample_vars, 'c_diff')
    ###        ic_default = old_ic_default + ic_cdiff_supplement
    ###
    ###        # setup and solve ODE
    ###        dose=1.0; duration=1.0; u_type='default'; u_params=[dose, duration, u_type]
    ###        t_end=200;
    ###        params_default = (mu_default, M_default, eps_default, u_params)
    ###        solve_vars = (ic_default, *params_default, t_end)
    ###        t, y = solve(solve_vars, solve_type='sporulation')
    ###
    ###        if_plot = 'save' # 'save' or 'show'
    ###        if_legend = True
    ###        legend_vars = t, y, labels, (if_legend, if_plot)
    ###        plot_standard(legend_vars, 'just_legend', 'new_figs/sporulation_legend.pdf')
    ###
    ###        print('total time elapsed is', time.clock() - tic0)
    ###    if example_type == 'basic':
    ###        # build parameters
    ###        var_data, ic_data = import_data()
    ###        labels, mu_default, M_default, eps_default = parse_data(var_data, \
    ###                model_type='default')
    ###        ic_num = 0 # initial conditions 0-8 available
    ###        old_ic_default = parse_ic((ic_data, ic_num), 'default')
    ###        [print(parse_ic((ic_data, i), 'default')) for i in range(9)]
    ####'IC ' + str(i) + ' has norm ' + 'a' + 
    #### save plot params to .filename.pdf

def save_metadata(filename, ic_type, solve_type, plot_type, ic_vars, sample_vars, solve_vars):
    import pickle
    data = {'filename':filename, 'ic_type':ic_type, 'solve_type':solve_type,
            'plot_type':plot_type, 'ic_vars':ic_vars, 'sample_vars':sample_vars,
            'solve_vars':solve_vars}
    prefix = 'new_figs/'
    meta_fname = filename[0:len(prefix)] + '.' + filename[len(prefix):]
    with open(meta_fname, 'wb') as handle:
        pickle.dump(data, handle)

# load plot params from .filename.pdf
def load_metadata(import_from):
    import pickle
    with open(import_from, 'rb') as handle:
        import_data = pickle.load(handle)
    import_data['filename']
    return import_data['filename'], import_data['ic_type'], import_data['solve_type'],\
     import_data['plot_type'], import_data['ic_vars'], import_data['sample_vars'],\
     import_data['solve_vars']

def multiplot(plot_varss, plot_types, filename, multiplot_type):
    import matplotlib.pyplot as plt
    #import seaborn as sns
    #sns.rcmod.reset_orig()
    #sns.set_palette(sns.color_palette("Set2", n_colors=12))
    if (multiplot_type == '2by1'):
        plt.clf()
        plt.subplot(2,1,1)
        plot_vars1, plot_vars2 = plot_varss

        plot_list_1 = list(plot_vars1)
        plot_list_1[-1] = 'multi'
        plot_vars1 = plot_list_1
        plot_list_2 = list(plot_vars2)
        plot_list_2[-1] = 'multi'
        plot_vars2 = plot_list_2

        plot_type1, plot_type2 = plot_types
        plot_standard(plot_vars1, plot_type1, filename)
        plt.subplot(2,1,2)
        plot_standard(plot_vars2, plot_type2, filename)
        plt.tight_layout()

        plt.savefig(filename)
        print('    saved', plot_types, 'to', filename, '    ')
    if (multiplot_type == 'solve_all_ics'):
        solve_vars, plot_vars_end = plot_varss; solve_type, plot_type = plot_types
        var_data, ic_data = import_data()
        for which_ic in range(9):
            plt.subplot(3,3,which_ic+1)
            print('solving ic', which_ic, end=', ')
            new_ic_condition = parse_ic((ic_data, which_ic), 'default')
            solve_vars[0] = new_ic_condition

            #ic, *params, t_end, sample, transplant_time = solve_vars
            t, y = solve(tuple(solve_vars), solve_type)
            #plot_vars = (y[-1],) + plot_vars_end
            plot_vars = (t, y) + plot_vars_end
            plot_standard(plot_vars, plot_type, filename)

        plt.savefig(filename)
        print('    saved', plot_type, 'for all ics to', filename, '    ')
    if (multiplot_type == 'solve_all_ics_normal'):
        solve_vars, plot_vars_end = plot_varss; solve_type, plot_type = plot_types
        var_data, ic_data = import_data()
        ic_standard = 4
        standard_size = sum(parse_ic((ic_data, ic_standard), 'default'))
        standard_size = 0.01
        for which_ic in range(9):
            plt.subplot(3,3,which_ic+1)
            #print('solving ic', which_ic, end=', ')
            base_ic_condition = parse_ic((ic_data, which_ic), 'default')
            new_ic_condition = base_ic_condition*(standard_size/sum(base_ic_condition))
            print('ic', which_ic, 'has size', sum(base_ic_condition))
            solve_vars[0] = new_ic_condition

            #ic, *params, t_end, sample, transplant_time = solve_vars
            t, y = solve(tuple(solve_vars), solve_type)
            #plot_vars = (y[-1],) + plot_vars_end
            plot_vars = (t, y) + plot_vars_end
            plot_standard(plot_vars, plot_type, filename)

        plt.savefig(filename)
        print('    saved', plot_type, 'for all ics to', filename, '    ')

def verbose_master():
    # choose scenario: 'default' or 'sporulation' or 'mutation_1' or 'mutation_2'
    master_scenario = 'default'
    # choose which initial condition to use (0-8 available)
    ic_num = 4

    # import parameters for gLV model
    global labels, mu, M, eps, ic_data
    var_data, ic_data  = import_data()
    labels, mu, M, eps = parse_data(var_data, scenario = master_scenario)
    ic                 = parse_ic((ic_data, ic_num), scenario = master_scenario)

    # choose parameters for first supplement
    # (supplement may be inoculation w/ C. diff, or transplant)
    global transplant_time_1, transplant_time_2, transplant_type_1, \
           transplant_type_2, transplant_strength_1, transplant_strength_2
    transplant_type_1   = 'cdiff' # 0-8 (corresponding to ICs), 'cdiff', or 'none'
    transplant_time_1   = 10; transplant_strength_1 = 1.0;
    transplant_vars_1   = [transplant_type_1, transplant_time_1, transplant_strength_1]

    # choose parameters for second supplement
    transplant_type_2   = 'cdiff' #'none' # 0-8 (corresponding to ICs), 'cdiff', or 'none'
    transplant_time_2   = 20; transplant_strength_2 = 1.0;
    transplant_vars_2   = [transplant_type_2, transplant_time_2, transplant_strength_2]

    # choose antibiotic treatment parameters
    global treatment_params
    treatment_type     = 'constant' # 'constant', 'tapered', or 'pulsed'
    treatment_dose     = 1.0; treatment_duration = 1.0
    treatment_params   = [treatment_type, treatment_dose, treatment_duration]

    # choose parameters for solving gLV equations
    # solve_type can be: 
    #   'microbe_trajectory' (microbe trajectories through time),
    #   'ss_u_transplant' (steady states, x-axis varies antibiotic, y-axis
    #                      amount of transplant),
    #   'ss_duration_dose' (steady states, x-axis varies treatment_duration,
    #                      y-axis treatment_dose)
    solve_type = 'microbe_trajectory'; t_end = 200;
    t, y       = master_solve(ic, t_end, master_scenario, solve_type)

    # choose parameters for plotting
    if_legend = True; if_plot='save'; filename='new_figs/m_plot.pdf';
    # solve type can be:
    #   'basic_traj_log' 
    #   'ss'
    plot_type='basic_traj_log';

    m_plot((t,y), (if_legend, if_plot), plot_type, filename, master_scenario)

def master(sc):
    # import parameters for gLV model
    global labels, mu, M, eps, ic_data
    var_data, ic_data  = import_data()
    labels, mu, M, eps = parse_data(var_data, scenario = sc.scenario_type)
    ic                 = parse_ic((ic_data, sc.ic_num), scenario = sc.scenario_type)
    if 'targeted' in sc.treatment_type:
        eps = np.zeros(len(mu))
        c_diff_index = labels.index("Clostridium difficile")
        eps[c_diff_index] = -1
        print(eps)
    t, y               = master_solve(ic, sc)
    points = m_plot((t,y), sc)
    print('     DONE in', time.clock() - tic0, 's')
    if points:
        return points
    else:
        return t, y

def make_figs(num):
    import matplotlib.pyplot as plt
    if num == 'u_options':
        t = np.linspace(0, 15, 1000)
        for u_type in ('constant', 'tapered', 'pulse'):
            total_dose = 2; duration = 10;
            y = [ u(tt, (u_type, total_dose, duration)) for tt in t]
            plt.axis([0, 10, 0, 1])
            if u_type == 'constant': linestyle = '-'
            if u_type == 'tapered': linestyle = ':'
            if u_type == 'pulse': linestyle = '--'
            plt.plot(t, y, linestyle=linestyle)
        plt.legend(['constant', 'tapered', 'pulsed'], fontsize=14)
        plt.xlabel('t (days)')
        plt.ylabel('antibiotic strength $u(t)$')
        plt.tight_layout()
        plt.savefig('new_figs/u_options_preplos.pdf', dpi=300)
    if num.startswith('0'):
        scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , False ,
                       'save'               , 'composition_snapshot' , 'new_figs/fig_'+num+'_preplos.pdf')
        master(scn)
    if num=='7':
        scn = Scenario('default'          , 4      ,
                       'cdiff'            , 10     , 1.0                         ,
                       7                  , 1, .1                         ,
                       'constant'         , 1.0    , 1.0                         ,
                       'microbe_trajectory' , 100    , False                       ,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t1,y1 = master(scn)

        scn = Scenario('default'          , 4      ,
                       'cdiff'            , 10     , 1.0                         ,
                       7                  , 7, .1                         ,
                       'constant'         , 1.0    , 1.0                         ,
                       'microbe_trajectory' , 100    , False                       ,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t2,y2 = master(scn)

        multi_scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , False ,
                       'save'               , '2_traj_plot' , 'new_figs/fig_'+num+'_preplos.pdf')
        m_plot(((t1, y1), (t2, y2)), multi_scn)

    if num=='1':
        scn = Scenario('default'            , 4                ,
                       'none'               , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 0.0              , 1.0   ,
                       'microbe_trajectory' , 150              , False ,
                       'save'               , 'basic_traj_log' , 'new_figs/fig_'+num+'.pdf')
        t1, y1 = master(scn)
        scn = Scenario('default'            , 4                ,
                       'none'               , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 150              , False ,
                       'save'               , 'basic_traj_log' , 'new_figs/fig_'+num+'.pdf')
        t2, y2 = master(scn)
        scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 0.71             , 1.0   ,
                       'microbe_trajectory' , 150              , False ,
                       'save'               , 'basic_traj_log' , 'new_figs/fig_'+num+'.pdf')
        t3, y3 = master(scn)
        scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 150              , False ,
                       'save'               , 'basic_traj_log' , 'new_figs/fig_'+num+'.pdf')
        t4, y4 = master(scn)

        multi_scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , False ,
                       'save'               , '4_traj_plot' , 'new_figs/fig_'+num+'_preplos.pdf')
        m_plot(((t1, y1), (t2, y2), (t3, y3), (t4, y4)), multi_scn)
        #m_plot(((t1, y1), (t1, y1), (t1, y1), (t1, y1)), multi_scn)
    if num=='1_legend':
        scn = Scenario('default'            , 4              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'microbe_trajectory' , 200            , False                       ,
                       'save'               , 'basic_legend' , 'new_figs/fig_'+num+'.pdf')
        master(scn)
    if num=='2':
        read_data = True
        import pickle
        if not read_data:
            x1,y1 = make_figs('2a')
            x2,y2 = make_figs('2b')
            x3,y3 = make_figs('2c')

            data = (x1,y1,x2,y2,x3,y3)
            with open('vars/reachable_ss.pi', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open('vars/reachable_ss.pi', 'rb') as f:
                x1,y1,x2,y2,x3,y3 = pickle.load(f)

        multi_scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , False ,
                       'save'               , '3_bw_ss' , 'new_figs/fig_'+num+'_preplos.eps')
        m_plot(((x1,y1), (x2,y2), (x3,y3)), multi_scn)

    if num=='2a':
        scn = Scenario('default'            , 0              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'ss_u_transplant'    , 500            , False                       ,
                       'save'               , 'ss_bw'           , 'new_figs/bw_fig_'+num+'.pdf')
        return master(scn)
    if num=='2b':
        scn = Scenario('default'            , 1              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'ss_u_transplant'    , 500            , False                       ,
                       'save'               , 'ss_bw'           , 'new_figs/bw_fig_'+num+'.pdf')
        return master(scn)
    if num=='2c':
        scn = Scenario('default'            , 4              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'ss_u_transplant'    , 500            , False                       ,
                       'save'               , 'ss_bw'           , 'new_figs/bw_fig_'+num+'.pdf')
        return master(scn)
    if num=='8_total':
        import pickle
        #read_data = True
        read_data = False
        if not read_data:
            x1,y1 = make_figs('8a')
            print(1,x1,y1)
            x2,y2 = make_figs('8b')
            print(2,x2,y2)
            x3,y3 = make_figs('8c')
            print(3,x3,y3)
            x4,y4 = make_figs('8d')
            print(4,x4,y4)
            x5,y5 = make_figs('8e')
            print(5,x5,y5)
            x6,y6 = make_figs('8f')
            print(6,x6,y6)

            data = (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6)
            with open('vars/transplant_data_donor_2.pi', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open('vars/transplant_data_donor_2.pi', 'rb') as f:
                x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6 = pickle.load(f)

        import matplotlib.pyplot as plt
        plt.clf()
        from scipy.interpolate import spline
        #x1 = x1[::-1]; x2 = x2[::-1]; x3 = x3[::-1]; x4 = x4[::-1]
        #y1 = y1[::-1]; y2 = y2[::-1]; y3 = y3[::-1]; y4 = y4[::-1]

        xx1 = np.linspace(x1[0], x1[-1], 300)
        xx2 = np.linspace(x2[0], x2[-1], 300)
        xx3 = np.linspace(x3[0], x3[-1], 300)
        xx4 = np.linspace(x4[0], x4[-1], 300)
        xx5 = np.linspace(x5[0], x5[-1], 300)
        xx6 = np.linspace(x6[0], x6[-1], 300)
        #sp2=spline(x2,y2,xx2)
        #print('sp2 is',sp2)
        #for (xx, yy, xxx) in ((x1,y1,xx1), (x2,y2,xx2), (x3,y3,xx3), (x4,y4,xx4), (x5,y5,xx5), (x6,y6,xx6)):
        fig, ax = plt.subplots()
        for (xx, yy, xxx) in ((x2,y2,xx2), (x4,y4,xx4), (x6,y6,xx6)):
            #try:
            #    plt.plot(xxx, spline(xx,yy,xxx),'k')
            #    print('spline is',spline(xx,yy,xxx))
            #except np.linalg.linalg.LinAlgError:
            #    plt.plot(xx, yy, 'k')
            #    print('vals are',xx,yy)
            ax.plot(xx, yy, 'k')

        #plt.plot(x1, y1, 'k', xx2, spline(x2,y2,xx2), 'k',
        #        xx3, spline(x3,y3,xx3),'k', xx4, spline(x4,y4,xx4),'k')
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels([0, 1, 2, 3, 4])
        ax.axis([0,2,0,1])
        ax.set_ylabel('transplant strength')
        ax.set_xlabel('antibiotic strength')
        fig.tight_layout()
        a1 = .57; a2 = 1.4; b1=.554; b2=.204
        ax.annotate("", xy=(a1, b1), xytext=(a2,b2),
                arrowprops=dict(arrowstyle='simple', facecolor='black',
                    edgecolor='none'))
        ax.text(.71,.5291, 'transplant delay', rotation=-29.6,
                bbox=dict(facecolor='white', edgecolor='white',lw=0))
        ax.text(.15, .05, 'uninfected', ha='left')
        ax.text(1.5, .05, 'infected', ha='left')
        fsize=16
        ax.text(1.32, .85, 'd=1', ha='left', fontsize=fsize)
        ax.text(1.125, .9, 'd=3', ha='left', fontsize=fsize)
        ax.text(.8, .9, 'd=5', ha='left', fontsize=fsize)
        ax.tick_params(axis='x', pad=10)

        plt.savefig('new_figs/fig_8_preplos.pdf', dpi=600)
    elif num.startswith('8'):
        if num=='8a': day = 0.001
        elif num=='8b': day=5
        elif num=='8c': day=4
        elif num=='8d': day=3
        elif num=='8e': day=2
        elif num=='8f': day=1
        scn = Scenario('default'          , 4      ,
                       'cdiff'            , 1     , 1.0                         ,
                       1                  , day , 1.0                         ,
                       'constant'         , 1.0    , 1.0                         ,
                       'ss_duration_dose' , 2000    , False                       ,
                       'save'             , 'ss_bw'   , 'new_figs/bw_fig_'+num+'.pdf')

        return master(scn)
    if num.startswith('3_total'):
        import pickle
        read_data = True
        #read_data = False
        if not read_data:
            x1,y1 = make_figs('3a')
            print(1,x1,y1)
            x2,y2 = make_figs('3b')
            print(2,x2,y2)
            x3,y3 = make_figs('3c')
            print(3,x3,y3)
            x4,y4 = make_figs('3d')
            print(4,x4,y4)
            x5,y5 = make_figs('3e')
            print(5,x5,y5)
            x6,y6 = make_figs('3f')
            print(6,x6,y6)

            data = (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6)
            with open('vars/transplant_data_2.pi', 'wb') as f:
                pickle.dump(data, f)
        else:
            with open('vars/transplant_data_2.pi', 'rb') as f:
                x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6 = pickle.load(f)

        import matplotlib.pyplot as plt
        plt.clf()
        from scipy.interpolate import spline
        #x1 = x1[::-1]; x2 = x2[::-1]; x3 = x3[::-1]; x4 = x4[::-1]
        #y1 = y1[::-1]; y2 = y2[::-1]; y3 = y3[::-1]; y4 = y4[::-1]

        xx1 = np.linspace(x1[0], x1[-1], 300)
        xx2 = np.linspace(x2[0], x2[-1], 300)
        xx3 = np.linspace(x3[0], x3[-1], 300)
        xx4 = np.linspace(x4[0], x4[-1], 300)
        xx5 = np.linspace(x5[0], x5[-1], 300)
        xx6 = np.linspace(x6[0], x6[-1], 300)
        #sp2=spline(x2,y2,xx2)
        #print('sp2 is',sp2)
        for (xx, yy, xxx) in ((x1,y1,xx1), (x2,y2,xx2), (x3,y3,xx3), (x4,y4,xx4), (x5,y5,xx5), (x6,y6,xx6)):
            #try:
            #    plt.plot(xxx, spline(xx,yy,xxx),'k')
            #    print('spline is',spline(xx,yy,xxx))
            #except np.linalg.linalg.LinAlgError:
            #    plt.plot(xx, yy, 'k')
            #    print('vals are',xx,yy)
            plt.plot(xx, yy, 'k')

        #plt.plot(x1, y1, 'k', xx2, spline(x2,y2,xx2), 'k',
        #        xx3, spline(x3,y3,xx3),'k', xx4, spline(x4,y4,xx4),'k')
        plt.axis([0,2,0,1])
        plt.ylabel('transplant strength')
        plt.xlabel('antibiotic strength')
        plt.tight_layout()
        ax = plt.axes()
        a1 = .67; a2 = 1.5; b1=.554; b2=.204
        ax.annotate("", xy=(a1, b1), xytext=(a2,b2),
                arrowprops=dict(arrowstyle='simple', facecolor='black',
                    edgecolor='none'))
        ax.text(.81,.5291, 'transplant delay', rotation=-30.75,
                bbox=dict(facecolor='white', edgecolor='white',lw=0))
        plt.text(.15, .05, 'uninfected', ha='left')
        plt.text(1.5, .05, 'infected', ha='left')
        fsize=16
        plt.text(1.7, .9, 'd=10', ha='left', fontsize=fsize)
        plt.text(1.7, .65, 'd=7', ha='left', fontsize=fsize)
        plt.text(1.7, .48, 'd=5', ha='left', fontsize=fsize)
        plt.text(1.7, .31, 'd=1', ha='left', fontsize=fsize)
        plt.text(1.1, .9, 'd=14', ha='left', fontsize=fsize)
        plt.text(.8, .9, 'd=30', ha='left', fontsize=fsize)
        plt.gca().tick_params(axis='x', pad=10)

        plt.savefig('new_figs/fig_3_preplos_2.pdf', dpi=600)
    elif num.startswith('3'):
        if num=='3a': day = 10.001
        elif num=='3b': day=14
        elif num=='3c': day=30
        elif num=='3d': day=5
        elif num=='3e': day=7
        elif num=='3f': day=1
        scn = Scenario('default'          , 4      ,
                       'cdiff'            , .99     , 1.0                         ,
                       7                  , day , 1.0                         ,
                       'constant'         , 1.0    , 1.0                         ,
                       'ss_duration_dose' , 2000    , False                       ,
                       'save'             , 'ss_bw'   , 'new_figs/bw_fig_'+num+'.pdf')

        return master(scn)
    if num=='4extra':
        scn = Scenario('mutation_1'          , 4      ,
                       'cdiff'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'constant'         , 10.0    , 100.0                         ,
                       'microbe_trajectory' , 300    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        master(scn)
    if num=='4':
        scn = Scenario('default'          , 'inf_ss'      ,
                       'none'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'constant_targeted'         , 10.0    , 30.0                         ,
                       'microbe_trajectory' , 100    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t1,y1 = master(scn)
        scn = Scenario('mutation_2'          , 'inf_ss'       ,
                       'none'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'constant_targeted'         , 10    , 30.0                         ,
                       'microbe_trajectory' , 100    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t2,y2 = master(scn)
        multi_scn = Scenario('mutation_2'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , True,
                       'save'               , '2_traj_plot' , 'new_figs/fig_'+num+'_preplos.pdf')
        #m_plot(((t1, y1), (t1, y1), (t1, y1), (t1, y1)), multi_scn)
        m_plot(((t1, y1), (t2, y2)), multi_scn)
    if num=='5_legend':
        scn = Scenario('default'            , 'inf_ss'              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'microbe_trajectory' , 200            , False                       ,
                       'save'               , 'basic_legend' , 'new_figs/fig_'+num+'.pdf')
        master(scn)
    if num=='4_legend':
        scn = Scenario('mutation_1'            , 4              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'microbe_trajectory' , 200            , False                       ,
                       'save'               , 'basic_legend' , 'new_figs/fig_'+num+'.pdf')
        master(scn)
    if num=='5':
        scn = Scenario('default'          , 'inf_ss'      ,
                       'none'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'pulse_targeted'         , 3    , 10.0                         ,
                       'microbe_trajectory' , 20    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t1,y1 = master(scn)
        scn = Scenario('default'          , 'inf_ss'      ,
                       'none'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'constant_targeted'         , 3    , 10.0                         ,
                       'microbe_trajectory' , 20    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t2,y2 = master(scn)
        scn = Scenario('sporulation'          , 'inf_ss'      ,
                       'none'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'pulse_targeted'         , 3    , 10.0                         ,
                       'microbe_trajectory' , 20    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t3,y3 = master(scn)
        scn = Scenario('sporulation'          , 'inf_ss'      ,
                       'none'            , 1     , 1.0                         ,
                       'none'                  , 11 , 1.0                         ,
                       'constant_targeted'         , 3    , 10.0                         ,
                       'microbe_trajectory' , 20    , False,
                       'save'             , 'basic_traj_log'   , 'new_figs/fig_'+num+'.pdf')
        t4,y4 = master(scn)
        multi_scn = Scenario('sporulation'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 200              , True,
                       'save'               , '4_traj_plot' , 'new_figs/fig_'+num+'_preplos.pdf')
        #m_plot(((t1, y1), (t1, y1), (t1, y1), (t1, y1)), multi_scn)
        m_plot(((t1, y1), (t2, y2), (t3, y3), (t4, y4)), multi_scn)
    if num=='5_legend':
        scn = Scenario('default'            , 'inf_ss'              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'microbe_trajectory' , 200            , False                       ,
                       'save'               , 'basic_legend' , 'new_figs/fig_'+num+'.pdf')
        master(scn)
    if num=='food_web':
        scn = Scenario('default'            , 4              ,
                       'cdiff'              , 10             , 1.0                         ,
                       'none'               , 20             , 1.0                         ,
                       'constant'           , 1.0            , 1.0                         ,
                       'microbe_trajectory' , 200            , False                       ,
                       'save'               , 'food_web' , 'new_figs/fig_'+num+'2.pdf')
        master(scn)
    if num.startswith('6'):
        scn = Scenario('default'            , 4                ,
                       'cdiff'              , 10               , 1.0   ,
                       'none'               , 20               , 1.0   ,
                       'constant'           , 1.0              , 1.0   ,
                       'microbe_trajectory' , 500              , False ,
                       'save'               , 'steady_state_snapshot' , 'new_figs/fig_'+num+'_preplos.pdf')
        master(scn)


Scenario = namedtuple('Scenario' , \
           ['scenario_type'     , 'ic_num'            ,
            'transplant_type_1' , 'transplant_time_1' , 'transplant_strength_1' ,
            'transplant_type_2' , 'transplant_time_2' , 'transplant_strength_2' ,
            'treatment_type'    , 'treatment_dose'    , 'treatment_duration'    ,
            'solve_type'        , 't_end'             , 'if_legend'             ,
            'if_plot'           , 'plot_type'         , 'filename'])

if __name__ == '__main__':
    print('this is a preliminary version of CDI_gLV_treatments.py')
    #example_plots('ss_from_infected')
    #example_plots('conc_vs_duration_multiplot_normalized')
    #example_plots('targeted_antibiotic_sporulation')
    #example_plots('show_vulnerable')
    #example_plots('plot_u')
    #example_plots('basic_mutation')
    #example_plots('show_vulnerable')
    ##example_plots('spor_w_transplant')
    ##example_plots('misc_transplants')
    #example_plots('master_example')
    #[make_figs(num) for num in ('1a', '1b', '1c', '1d')]
#    make_figs('1_legend') # remember to pdfcrop this image
#    [make_figs(num) for num in ('2a', '2b', '2c')] # fontsize 28
#    [make_figs(num) for num in ('7a', '7b', '7c', '7d')] # fontsize 28
    #[make_figs(num) for num in ('3a', '3b', '3c', '3d')] # fontsize 28
    #make_figs('3_total')
    #make_figs('6')
    #make_figs('0b')
##    make_figs('4a')
##    make_figs('4b')
    #make_figs('3e')
    #[make_figs(num) for num in ('5a', '5b', '5c', '5d')]
    #make_figs('5b')
    #make_figs('5_legend')
    #make_figs('4_legend')
    #make_figs('u_options')

# FINAL FIGS
    #make_figs('food_web')
    #make_figs('0') # font size 20
    #make_figs('1') # font size 18
    #make_figs('2') # font size 14
    #make_figs('3_total_2') # font size 20
    #make_figs('7') # font size 18
    #make_figs('5')
    #make_figs('4') # font size 18
    #make_figs('4_legend')
    #make_figs('6') # font size 24
    #make_figs('u_options') # font size 20
    #make_figs('8_total')
