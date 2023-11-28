import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2023)
pred = []
# Number of functions
for i in range(100):


    #alphas = [1.5,2,2.5]
    alphas = [1.8,2.2,2.9]

    
    for alpha in alphas:
        
        N = 100
        k = 3

        xis = np.random.standard_normal(size = N)
        lmbds = np.array([1/(j**alpha) for j in range(1,N+1)])
        coefs = xis*lmbds
        x_test = np.linspace(0,1,100)

        sine_matrix = np.array([np.sin(j*np.pi*x_test) for j in range(N)])

        y_pred_test = sine_matrix.T @ coefs

        pred.append(y_pred_test)

pred = np.array(pred)

print('Number of functions: {}'.format(int(pred.shape[0])))
print('Number of points per function: {}'.format(pred.shape[1]))

np.save('burgers_ics.npy',pred)


# alphas repeated
alphas_full = np.tile(alphas,pred.shape[0])
# Plotting the solution
plt.figure(figsize=(10, 6))
for i in range(pred.shape[0]):
    plt.plot(x_test,pred[i,:], label = '{}'.format(alphas_full[i]))
plt.title('Initial conditions for Burgers equation')
#plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)

# Save the plot to a PNG file
plt.savefig('burgers_ics.png')

plt.show()