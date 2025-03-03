import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from scipy.spatial import distance
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
#    import sobol_seq
import os
import jax.numpy as jnp
import sys




# Neural network 
# def get_burgers_coeffs_pinnsr(n_obs,run):
with tf.device('/device:GPU:0'):
# =============================================================================
#  define loss histories to record convergence
# =============================================================================
    # L-BFGS-S loss history
    # loss_history = np.array([0])
    # loss_u_history = np.array([0])
    # loss_f_history = np.array([0])
    # loss_lambda_history = np.array([0])
    # loss_history_val = np.array([0])
    # loss_u_history_val = np.array([0])
    # loss_f_history_val = np.array([0])
    # step = 0
    
    # Adam loss history
    loss_history_Adam = np.array([0])
    loss_u_history_Adam = np.array([0])
    loss_f_history_Adam = np.array([0])
    loss_lambda_history_Adam = np.array([0])
    lambda_history_Adam = np.zeros((10,1))  
    loss_history_Adam_val = np.array([0])
    loss_u_history_Adam_val = np.array([0])
    loss_f_history_Adam_val = np.array([0])
    
    # STRidge loss history
    loss_history_STRidge = np.array([0])
    loss_f_history_STRidge = np.array([0])
    loss_lambda_history_STRidge = np.array([0])
    optimaltol_history = np.array([0])   
    tol_history_STRidge = np.array([0])
    lambda_normalized_history_STRidge = np.zeros((10,1))
    
    lambda_history_STRidge = np.zeros((10,1))
    ridge_append_counter_STRidge = np.array([0])
    
    # Loss histories for pretraining
    loss_history_Pretrain = np.array([0])
    loss_u_history_Pretrain = np.array([0])
    loss_f_history_Pretrain = np.array([0])
    loss_lambda_history_Pretrain = np.array([0])
    loss_history_val_Pretrain = np.array([0])
    loss_u_history_val_Pretrain = np.array([0])
    loss_f_history_val_Pretrain = np.array([0])
    step_Pretrain = 0
    
    lambda_history_Pretrain = np.zeros((10,1))  
    
    np.random.seed(1234)
    tf.set_random_seed(1234)
    
    class PhysicsInformedNN:
# =============================================================================
#     Inspired by Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis.
#     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
#     involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
# =============================================================================
        # Initialize the class
        def __init__(self, X, u, X_f, X_val, u_val, layers, lb, ub):
            
            self.lb = lb
            self.ub = ub
            self.layers = layers
            
            # Initialize NNs
            self.weights, self.biases = self.initialize_NN(layers)
            
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            
            # Initialize parameters
            self.lambda1 = tf.Variable(tf.zeros([10, 1], dtype=tf.float32), dtype=tf.float32, name = 'lambda')
            
            # Specify the list of trainable variables 
            var_list_1 = self.biases + self.weights
            
            var_list_Pretrain = self.biases + self.weights
            var_list_Pretrain.append(self.lambda1)
            
            ######### Training data ################
            self.x = X[:,0:1]
            self.t = X[:,1:2]
            self.u = u
            self.x_f = X_f[:, 0:1]
            self.t_f = X_f[:, 1:2]
            
            self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
            self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
            self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
            self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
            
            self.u_pred = self.net_u(self.x_tf, self.t_tf)
            self.f_pred, self.Phi_pred, self.u_t_pred = self.net_f(self.x_f_tf, self.t_f_tf, self.x_f.shape[0])
            
            self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
            self.loss_f_coeff_tf = tf.placeholder(tf.float32)
            self.loss_f = self.loss_f_coeff_tf*tf.reduce_mean(tf.square(self.f_pred))
            
            self.loss_lambda = 1e-7*tf.norm(self.lambda1, ord = 1)       
            
            
            self.loss = tf.log(self.loss_u  + self.loss_f + self.loss_lambda) # log loss
            
            ######### Validation data ################
            self.x_val = X_val[:,0:1]
            self.t_val = X_val[:,1:2]
            self.u_val = u_val
            
            self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
            self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
            self.u_val_tf = tf.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])
                    
            self.u_val_pred = self.net_u(self.x_val_tf, self.t_val_tf)
            self.f_val_pred, _, _ = self.net_f(self.x_val_tf, self.t_val_tf, self.x_val.shape[0])
            
            self.loss_u_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
            self.loss_f_val = tf.reduce_mean(tf.square(self.f_val_pred))
            self.loss_val = tf.log(self.loss_u_val  + self.loss_f_val) # log loss
                        
            ######### Optimizor #########################
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    var_list = var_list_1,
                                                                    method = 'L-BFGS-B', 
                                                                    options = {'maxiter': 1000,
                                                                            'maxfun': 1000,
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps})
    
            self.optimizer_Pretrain = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                    var_list = var_list_Pretrain,
                                                                    method = 'L-BFGS-B', 
                                                                options = {'maxiter': 10000,
                                                                            'maxfun': 10000,
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps})
                                
            # The default settings: learning rate = 1e-3, beta1 = 0.9, beta2 = 0.999， epsilon = 1e-8
            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 1e-3
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.75,
                                                        staircase=True)
            self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3, beta1 = 0.99, beta2 = 0.9, epsilon = 1e-8) 
            self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list = var_list_1,
                                                            global_step = self.global_step)
            
            self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u, 
                    self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                    self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.u_val_tf: self.u_val}
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
    
        def initialize_NN(self, layers):        
            weights = []
            biases = []
            num_layers = len(layers) 
            for l in range(0,num_layers-1):
                W = self.xavier_init(size=[layers[l], layers[l+1]])
                b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name = 'b')
                weights.append(W)
                biases.append(b)        
            return weights, biases
            
        def xavier_init(self, size):
            in_dim = size[0]
            out_dim = size[1]        
            xavier_stddev = np.sqrt(2/(in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32, name = 'W')
        
        def neural_net(self, X, weights, biases):
            num_layers = len(weights) + 1
            
            H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
            for l in range(0,num_layers-2):
                W = weights[l]
                b = biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = weights[-1]
            b = biases[-1]
            Y = tf.add(tf.matmul(H, W), b)
            return Y
                
        def net_u(self, x, t):  
            u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
            return u
        
        def net_f(self, x, t, N_f):
            u = self.net_u(x,t)
            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_xx = tf.gradients(u_x, x)[0]
            u_xxx = tf.gradients(u_xx, x)[0]    
            Phi = tf.concat([tf.constant(1, shape=[N_f, 1], dtype=tf.float32),
                             u,
                             u_x,
                             u_xx,
                             u**2,
                             u*u_x,
                             u*u_xx,
                             u_x*u_x,
                             u_x*u_xx,
                             u_xx*u_xx], 1)    
            self.library_description = ['1',
                                        'u',
                                        'u_x',
                                        'u_xx',
                                        'u**2',
                                        'u*u_x',
                                        'u*u_xx',
                                        'u_x*u_x',
                                        'u_x*u_xx',
                                        'u_xx*u_xx']
            
            f = tf.matmul(Phi, self.lambda1) - u_t      
            return f, Phi, u_t
        
        # def callback(self, loss, loss_u, loss_f, loss_lambda, loss_val, loss_u_val, loss_f_val):
        #     global step
        #     step = step+1
        #     if step%10 == 0:
        #         print('Step: %d, log Loss: %e, loss_u: %e, loss_f: %e, loss_lambda: %e' % (step, loss, loss_u, loss_f, loss_lambda))
                
        #         global loss_history
        #         global loss_u_history
        #         global loss_f_history
        #         global loss_lambda_history
                
        #         global loss_history_val
        #         global loss_u_history_val
        #         global loss_f_history_val
                
        #         loss_history = np.append(loss_history, loss)
        #         loss_u_history = np.append(loss_u_history, loss_u)
        #         loss_f_history = np.append(loss_f_history, loss_f)
        #         loss_lambda_history = np.append(loss_lambda_history, loss_lambda)
                
        #         loss_history_val = np.append(loss_history_val, loss_val)
        #         loss_u_history_val = np.append(loss_u_history_val, loss_u_val)
        #         loss_f_history_val = np.append(loss_f_history_val, loss_f_val)
                
        def callback_Pretrain(self, loss, loss_u, loss_f, loss_lambda, loss_val, loss_u_val, loss_f_val, lamu):
            global step_Pretrain
            step_Pretrain += 1
            if step_Pretrain % 10 == 0:
                print('Step: %d, log Loss: %e, loss_u: %e, loss_f: %e, loss_lambda: %e' % (step_Pretrain, loss, loss_u, loss_f,
                                                                                        loss_lambda))
                
                global loss_history_Pretrain
                global loss_u_history_Pretrain
                global loss_f_history_Pretrain
                global loss_lambda_history_Pretrain
                
                global loss_history_val_Pretrain
                global loss_u_history_val_Pretrain
                global loss_f_history_val_Pretrain
                
                global lambda_history_Pretrain
                
                loss_history_Pretrain = np.append(loss_history_Pretrain, loss)
                loss_u_history_Pretrain = np.append(loss_u_history_Pretrain, loss_u)
                loss_f_history_Pretrain = np.append(loss_f_history_Pretrain, loss_f)
                loss_lambda_history_Pretrain = np.append(loss_lambda_history_Pretrain, loss_lambda)
                
                loss_history_val_Pretrain = np.append(loss_history_val_Pretrain, loss_val)
                loss_u_history_val_Pretrain = np.append(loss_u_history_val_Pretrain, loss_u_val)
                loss_f_history_val_Pretrain = np.append(loss_f_history_val_Pretrain, loss_f_val)
                
                lambda_history_Pretrain = np.append(lambda_history_Pretrain, lamu, axis = 1)
            
        def train(self, nIter): # nIter is the number of ADO loop
            self.tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u, 
                    self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                    self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.u_val_tf: self.u_val,
                    self.loss_f_coeff_tf: 1}    
            
            # Pretraining,as a form of a good intialization
            print('L-BFGS-B pretraining begins')
            self.optimizer_Pretrain.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss, self.loss_u, self.loss_f, self.loss_lambda,
                                            self.loss_val, self.loss_u_val, self.loss_f_val, self.lambda1],
                                    loss_callback = self.callback_Pretrain)
            
            self.tf_dict[self.loss_f_coeff_tf] = 2
            for self.it in range(nIter):
                
                # Loop of STRidge optimization
                print('STRidge begins')
                self.callTrainSTRidge()
    
                # Loop of Adam optimization
                print('Adam begins')
                start_time = time.time()
                for it_Adam in range(1000):
                    
                    self.sess.run(self.train_op_Adam, self.tf_dict)
                    
                    # Print
                    if it_Adam % 10 == 0:
                        elapsed = time.time() - start_time
                        loss, loss_u, loss_f, loss_lambda, lambda1_value, loss_val, loss_u_val, loss_f_val = self.sess.run([self.loss, self.loss_u, self.loss_f, self.loss_lambda, self.lambda1, self.loss_val, self.loss_u_val, self.loss_f_val], self.tf_dict)
                        print('It: %d, Log Loss: %.3e, loss_u: %e, loss_f: %e, loss_lambda: %e, Time: %.2f' 
                            % (it_Adam, loss, loss_u, loss_f, loss_lambda, elapsed))
                        
                        lamu = self.sess.run(self.lambda1)
                        
                        global loss_history_Adam
                        global lambda_history_Adam
                        global loss_u_history_Adam
                        global loss_f_history_Adam
                        global loss_lambda_history_Adam
                        
                        global loss_history_Adam_val
                        global loss_u_history_Adam_val
                        global loss_f_history_Adam_val
                        
                        loss_history_Adam = np.append(loss_history_Adam, loss)
                        lambda_history_Adam = np.append(lambda_history_Adam, lambda1_value, axis=1)
                        loss_u_history_Adam = np.append(loss_u_history_Adam, loss_u)
                        loss_f_history_Adam = np.append(loss_f_history_Adam, loss_f)
                        loss_lambda_history_Adam = np.append(loss_lambda_history_Adam, loss_lambda)
                        
                        loss_history_Adam_val = np.append(loss_history_Adam_val, loss_val)
                        loss_u_history_Adam_val = np.append(loss_u_history_Adam_val, loss_u_val)
                        loss_f_history_Adam_val = np.append(loss_f_history_Adam_val, loss_f_val)
                        
                        lambda_history_Adam = np.append(lambda_history_Adam, lamu, axis = 1)
                
                        start_time = time.time()
                
                # Loop of L-BFGS-B optimization
                # print('L-BFGS-B begins')
                # self.optimizer.minimize(self.sess,
                #                         feed_dict = self.tf_dict,
                #                         fetches = [self.loss, self.loss_u, self.loss_f, self.loss_lambda,
                #                                    self.loss_val, self.loss_u_val, self.loss_f_val],
                #                         loss_callback = self.callback)
                
            # one more time of STRidge optimization
            print('STRidge begins')
            self.callTrainSTRidge()
    
    
        def predict(self, X_star):
            
            tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
            
            u_star = self.sess.run(self.u_pred, tf_dict)
            
            return u_star
        
        def callTrainSTRidge(self):
            lam = 1e-5
            d_tol = 1
            maxit = 100
            STR_iters = 10
            
            l0_penalty = None
            
            normalize = 2
            split = 0.8
            print_best_tol = False     
            Phi_pred, u_t_pred = self.sess.run([self.Phi_pred, self.u_t_pred], self.tf_dict)
            
            lambda2 = self.TrainSTRidge(Phi_pred, u_t_pred, lam, d_tol, maxit, STR_iters, l0_penalty, normalize, split,
                                            print_best_tol)     
            
            self.lambda1 = tf.assign(self.lambda1, tf.convert_to_tensor(lambda2, dtype = tf.float32))
                    
        def TrainSTRidge(self, R0, Ut, lam, d_tol, maxit, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, 
                        print_best_tol = False):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================           
            # First normalize data 
            n,d = R0.shape
            R = np.zeros((n,d), dtype=np.float32)
            if normalize != 0:
                Mreg = np.zeros((d,1))
                for i in range(0,d):
                    Mreg[i] = 1.0/(np.linalg.norm(R0[:,i],normalize))
                    R[:,i] = Mreg[i]*R0[:,i]                
                normalize_inner = 0
            else: 
                R = R0
                Mreg = np.ones((d,1))*d
                normalize_inner = 2
            
        
            global lambda_normalized_history_STRidge
            lambda_normalized_history_STRidge = np.append(lambda_normalized_history_STRidge, Mreg, axis = 1)
            
            # Split data into 80% training and 20% test, then search for the best tolderance.
            np.random.seed(0) # for consistancy
            n,_ = R.shape
            train = np.random.choice(n, int(n*split), replace = False)
            test = [i for i in np.arange(n) if i not in train]
            TrainR = R[train,:]
            TestR = R[test,:]
            TrainY = Ut[train,:]
            TestY = Ut[test,:]
        
            # Set up the initial tolerance and l0 penalty
            d_tol = float(d_tol)
            if self.it == 0: 
                self.tol = d_tol
                    
            # Or inherit Lambda
            w_best = self.sess.run(self.lambda1)/Mreg
            
            # err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
            err_f = np.mean((TestY - TestR.dot(w_best))**2)
            
            if l0_penalty == None and self.it == 0: 
                self.l0_penalty_0 = err_f
                l0_penalty = self.l0_penalty_0
            elif l0_penalty == None:
                l0_penalty = self.l0_penalty_0            
                
            err_lambda = l0_penalty*np.count_nonzero(w_best)
            err_best = err_f + err_lambda
            tol_best = 0
                        
            global loss_history_STRidge
            global loss_f_history_STRidge
            global loss_lambda_history_STRidge
            global tol_history_STRidge
            
            loss_history_STRidge = np.append(loss_history_STRidge, err_best)
            loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
            loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
            tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
        
            # Now increase tolerance until test performance decreases
            for iter in range(maxit):
        
                # Get a set of coefficients and error
                w = self.STRidge(TrainR, TrainY, lam, STR_iters, self.tol, Mreg, normalize = normalize_inner)
                
                # err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
                err_f = np.mean((TestY - TestR.dot(w))**2)
                
                err_lambda = l0_penalty*np.count_nonzero(w)
                err = err_f + err_lambda
        
                # Has the accuracy improved?
                if err <= err_best:
                    err_best = err
                    w_best = w
                    tol_best = self.tol
                    self.tol = self.tol + d_tol
                    
                    loss_history_STRidge = np.append(loss_history_STRidge, err_best)
                    loss_f_history_STRidge = np.append(loss_f_history_STRidge, err_f)
                    loss_lambda_history_STRidge = np.append(loss_lambda_history_STRidge, err_lambda)
                    tol_history_STRidge = np.append(tol_history_STRidge, tol_best)
        
                else:
                    self.tol = max([0,self.tol - 2*d_tol])
                    d_tol = d_tol/1.618
                    self.tol = self.tol + d_tol
        
            if print_best_tol: print ("Optimal tolerance:", tol_best)
            
            global optimaltol_history
            optimaltol_history = np.append(optimaltol_history, tol_best)
                    
            return np.real(np.multiply(Mreg, w_best))     
        
        def STRidge(self, X0, y, lam, maxit, tol, Mreg, normalize = 2, print_results = False):
        
            n,d = X0.shape
            X = np.zeros((n,d), dtype=np.complex64)
            # First normalize data
            if normalize != 0:
                Mreg = np.zeros((d,1))
                for i in range(0,d):
                    Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
                    X[:,i] = Mreg[i]*X0[:,i]                
            else: 
                X = X0
            
            # Inherit lambda
            w = self.sess.run(self.lambda1)/Mreg            
            
            biginds = np.where(abs(w) > tol)[0]
            num_relevant = d            
            
            global ridge_append_counter_STRidge
            ridge_append_counter = 0
            
            global lambda_history_STRidge
            lambda_history_STRidge = np.append(lambda_history_STRidge, np.multiply(Mreg, w), axis = 1)
            ridge_append_counter += 1
            
            # Threshold and continue
            for j in range(maxit):
        
                # Figure out which items to cut out
                smallinds = np.where(abs(w) < tol)[0]
                new_biginds = [i for i in range(d) if i not in smallinds]
                    
                # If nothing changes then stop
                if num_relevant == len(new_biginds): break
                else: num_relevant = len(new_biginds)
                    
                if len(new_biginds) == 0:
                    if j == 0: 
                        if normalize != 0:
                            lambda_history_STRidge = np.append(lambda_history_STRidge, w*Mreg, axis = 1)
                            ridge_append_counter += 1
                            ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                            return np.multiply(Mreg, w)
                        else:
                            lambda_history_STRidge = np.append(lambda_history_STRidge, w*Mreg, axis = 1)
                            ridge_append_counter += 1
                            ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                            return w
                    else: break
                biginds = new_biginds
                
                # Otherwise get a new guess
                w[smallinds] = 0
                
                if lam != 0: 
                    w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
                    lambda_history_STRidge = np.append(lambda_history_STRidge, np.multiply(Mreg,w), axis = 1)
                    ridge_append_counter += 1
                else: 
                    w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
                    lambda_history_STRidge = np.append(lambda_history_STRidge, np.multiply(Mreg,w), axis = 1)
                    ridge_append_counter += 1
        
            # Now that we have the sparsity pattern, use standard least squares to get w
            if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
            
            if normalize != 0:
                lambda_history_STRidge = np.append(lambda_history_STRidge, w*Mreg, axis = 1)
                ridge_append_counter += 1
                ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                return np.multiply(Mreg, w)
            else:
                lambda_history_STRidge = np.append(lambda_history_STRidge, w*Mreg, axis = 1)
                ridge_append_counter += 1
                ridge_append_counter_STRidge = np.append(ridge_append_counter_STRidge, ridge_append_counter)
                return w
    
    if __name__ == "__main__":                 

        start_time = time.time()

        layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

        # Read data
        
        data_for_pinn =  np.load('data_in_PINNSR.npy', allow_pickle=True)
        n_obs = int(sys.argv[1])
        run = int(sys.argv[2])
        for element in data_for_pinn:
            if element[0] == n_obs and element[1] == run:
                data_input = element[2]
                print(data_input)

        # unpack elements to train                
        X_u_train, u_train, X_f_train, X_u_val, u_val, lb, ub = data_input
        
        # train
        model = PhysicsInformedNN(X_u_train, u_train.reshape(-1,1), X_f_train, X_u_val, u_val.reshape(-1,1), layers, lb, ub)
        model.train(6)
        # Predict in fine grid
        num_fine_grid = 100
        t_fine,x_fine = np.meshgrid(np.linspace(0,1,num_fine_grid+4)[2:-2],np.linspace(0,1,num_fine_grid+4)[2:-2])
        tx_fine_int = np.vstack([t_fine.flatten(),x_fine.flatten()]).T
        tx_fine_int[:,[1,0]] = tx_fine_int[:,[0,1]] # Final swap
        u_pred = model.predict(tx_fine_int)

        # Get coefficients
        coeffs = model.sess.run(model.lambda1)
        print(coeffs)
        
        # load array
        data_out_PINNSR = np.load('data_out_PINNSR.npy', allow_pickle=True)
        
        # Append coeffs and u_pred
        for element in data_out_PINNSR:
            if element[0] == n_obs and element[1] == run:
                element[2] = coeffs
                element[3] = u_pred
        # save array
        np.save(f'data_out_PINNSR.npy', np.array(data_out_PINNSR, dtype=object), allow_pickle=True)        
                    
             
        
    



