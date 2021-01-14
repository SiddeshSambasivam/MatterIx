import math
import sys
print(sys.path)
import modules.activations

def prob_traffic(capacity, active_percent, thres=10):
    '''
    Each User is active only for active_perccent % of the time
    '''
    prob = 0
    assert active_percent <=1
    for i in range(thres+1, capacity+1):
        fact = math.factorial(capacity)/(math.factorial(i)*math.factorial(capacity-i))
        lprob = (active_percent)**i * (1-active_percent)**(capacity-i)
        prob+= (fact * lprob)
    print()
    print(f'The network can support {capacity} users. Given each user os active {active_percent*100}% of time.')
    print(f'The probability for more than {thres} users to be active at the same time= {prob}')
    print()

if __name__ == "__main__":
    prob_traffic(35, 0.1)
    prob_traffic(100, 0.1)



