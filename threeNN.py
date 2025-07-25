import sys
def main():
    #y = 2x+0.3
    # x0 corresponds to the x coordinate, x1 corresponds to the y coordinate of a point
    # If the given point is below the y = 2x + 0.3 line, the Neural Network is supposed
    # to output a 0, if the point is above the line, it's output is to be 1.
    #---create training data---
    x0 = [1,2,3,4,5,6,7,8,9,10]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23,16.2,18.4,20.4]
    y = [0,1,0,1,0,1,0,0,1,1] #expected output of the network

    #---initialise the weights and biases---
    w0 = 0.1
    w1 = 0.4
    w2 = -0.2
    w3 = 0.7
    w4 = -0.1
    w5 = 0.5
    b0 = 0.22
    b1 = -0.1
    b2 = 0.3
    #---train the single NN---
    #need 10000 epochs, with 0.001 learning rate
    for i in range(0,10000):
        loss = 0
        for j in range(0,len(y)):
            #forward pass
            s0 = x0[j]*w0 + x1[j]*w1 + b0
            s1 = x0[j]*w2 + x1[j]*w3 + b1
            a0 = s0
            a1 = s1
            s2 = a0*w4 + a1*w5 + b2
            a2 = s2

            loss= loss+0.5*(y[j]-a2)**2 #compute loss
            dw4 = -(y[j]-a2)*a0 #compute gradients
            dw5 = -(y[j]-a2)*a1
            db2 = -(y[j]-a2)
            dw0 = -(y[j]-a2)*w4*x0[j]
            dw1 = -(y[j]-a2)*w4*x1[j]
            dw2 = -(y[j]-a2)*w5*x0[j]
            dw3 = -(y[j]-a2)*w5*x1[j]
            db0 = -(y[j]-a2)*w4
            db1 = -(y[j]-a2)*w5
            #update weights, biases
            w0 = w0 - 0.001 * dw0
            w1 = w1 - 0.001 * dw1
            w2 = w2 - 0.001 * dw2
            w3 = w3 - 0.001 * dw3
            w4 = w4 - 0.001 * dw4
            w5 = w5 - 0.001 * dw5
            b0 = b0 - 0.001 * db0
            b1 = b1 - 0.001 * db1
            b2 = b2 - 0.001 * db2
        print('loss =',loss)

    #---test for unkown data, on the trained network---
    x0 = 2.7 #x coord. of point
    x1 = 6.0 #y coord. of point
    s0 = x0*w0 + x1*w1 + b0
    s1 = x0*w2 + x1*w3 + b1
    a0 = s0
    a1 = s1
    s2 = w4*a0 + w5*a1 + b2
    a2 = s2
    print('output for(',x0, ',', x1,')=' ,a2)

    x0 = 5.3 #x coord. of point
    x1 = 10.4 #y coord. of point
    s0 = x0*w0 + x1*w1 + b0
    s1 = x0*w2 + x1*w3 + b1
    a0 = s0
    a1 = s1
    s2 = w4*a0 + w5*a1 + b2
    a2 = s2
    print('output for(',x0, ',', x1,')=' ,a2)


if __name__ == '__main__':
    sys.exit(int(main() or 0))