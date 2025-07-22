import sys

def main():
    #y = 2x+0.3
    # x0 corresponds to the x coordinate, x1 corresponds to the y coordinate of a point
    # If the given point is below the y = 2x + 0.3 line, the Neural Network is supposed
    # to output a 0, if the point is above the line, it's output is to be 1.
    #---create some training data---
    x0 = [1,2,3,4,5,6,7,8,9,10]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23,16.2,18.4,20.4]
    y = [0,1,0,1,0,1,0,0,1,1] 

    #---initialise weights and biases---
    w0 = 0.1
    w1 = 0.4
    w2 = -0.2
    w3 = 0.7
    w4 = -0.1
    w5 = 0.5
    w6 = 0.2
    w7 = -0.4
    w8 = 0.6
    w9 = 0.1
    b0 = 0.22
    b1 = -0.1
    b2 = 0.3
    b3 = -0.3
    b4 = 0.2
    #---train the single NN---
    #need 10000 epochs, with 0.001 learning rate
    for i in range(0,10000):
        loss = 0
        for j in range(0,len(y)):
            #forward pass
            s0 = x0[j]*w0 + x1[j]*w1 +b0
            s1 = x0[j]*w2 + x1[j]*w3 +b1
            a0 = s0
            a1 = s1
            s2 = a0*w4 + a1*w5 + b2
            s3 = a0*w6 + a1*w7 + b3
            a2 = s2
            a3 = s3
            s4 = a2*w8 + a3*w9 + b4
            a4 = s4
            
            loss = loss + 0.5*(y[j]-a4)**2

            #delta
            delta4 = -(y[j]-a4)
            delta3 = delta4*w9
            delta2 = delta4*w8
            delta1 = delta3*w7 + delta2*w5
            delta0 = delta3*w6 + delta2*w4

            #compute gradient
            dw0 = delta0 * x0[j]
            dw1 = delta0 * x1[j]
            dw2 = delta1 * x0[j]
            dw3 = delta1 * x1[j]
            dw4 = delta2 * a0
            dw5 = delta2 * a1
            dw6 = delta3 * a0
            dw7 = delta3 * a1
            dw8 = delta4 * a2
            dw9 = delta4 * a3
            db0 = delta0
            db1 = delta1
            db2 = delta2
            db3 = delta3
            db4 = delta4

            #update weights and biases 
            w0 = w0 - 0.001 * dw0
            w1 = w1 - 0.001 * dw1
            w2 = w2 - 0.001 * dw2
            w3 = w3 - 0.001 * dw3
            w4 = w4 - 0.001 * dw4
            w5 = w5 - 0.001 * dw5
            w6 = w6 - 0.001 * dw6
            w7 = w7 - 0.001 * dw7
            w8 = w8 - 0.001 * dw8
            w9 = w9 - 0.001 * dw9
            b0 = b0 - 0.001 * db0
            b1 = b1 - 0.001 * db1
            b2 = b2 - 0.001 * db2
            b3 = b3 - 0.001 * db3
            b4 = b4 - 0.001 * db4
        print('loss = ', loss)

    #---test for unkown data, on the trained network---
    x0 = 2.7 # x coord. of point
    x1 = 6.0 # y coord. of point
    s0 = x0 * w0 + x1 * w1 + b0
    s1 = x0 * w2 + x1 * w3 + b1
    a0 = s0
    a1 = s1
    s2 = a0 * w4 + a1 * w5 + b2
    a2 = s2
    s3 = a0 * w6 + a1 * w7 + b3
    a3 = s3
    s4 = a2 * w8 + a3 * w9 + b4
    a4 = s4
    print('output for (',x0,',',x1,')= ',a4)
    x0 = 5.3 # x coord. of point
    x1 = 10.4 # y coord. of point
    s0 = x0 * w0 + x1 * w1 + b0
    s1 = x0 * w2 + x1 * w3 + b1
    a0 = s0
    a1 = s1
    s2 = a0 * w4 + a1 * w5 + b2
    a2 = s2
    s3 = a0 * w6 + a1 * w7 + b3
    a3 = s3
    s4 = a2 * w8 + a3 * w9 + b4
    a4 = s4
    print('output for (',x0,',',x1,')= ',a4)
    

if __name__ == '__main__':
    sys.exit(int(main() or 0))