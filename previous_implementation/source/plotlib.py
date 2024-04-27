import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

def mse_plot(x, true, pred, title):

    mse_u = mean_squared_error(true,pred)/mean_squared_error(0*true,true)

    fig = plt.figure()
    plt.plot(x, pred,label = 'pred', marker= 'o')
    plt.plot(x, true, label = 'true', marker = 'o')
    plt.title(title)
    plt.legend()    
    print('{} = {}'.format('Normalized MSE : ',mse_u))
    return fig