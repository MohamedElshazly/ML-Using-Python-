
from numpy import *


#this function is to compute the length between the the points of our data set and the linear regression line that we are trying to make it fit the data well 
def compute_error_for_given_points(b,m,points):
	totalError= 0 
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError +=(y - (m*x + b))**2
	return totalError /  float(len(points))	


#this function is the implementation of  the cost function and gradient descent (theta:= theta - alpha*(1/2N *sum(sqrtError)))
def step_gradient(b_current , m_current , points, learningRate):
	b_gradient = 0 
	m_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		 x = points[i,0]
		 y = points[i,1]
		 b_gradient += -(2/N) * (y - ((m_current*x) + b_current))
		 m_gradient += -(2/N) * x * (y - ((m_current*x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]	 




# we are updating the values of b and m 
def gradient_descent_runner(points , starting_b , starting_m , learning_rate , num_iteration ):
	 b = starting_b 
	 m = starting_m

	 for i in range(num_iteration):
	 	b ,m = step_gradient(b , m , array(points) , learning_rate)
	 return [b,m]	

#we print our final results 
def run():
	points= genfromtxt('data.csv',delimiter=',')
	#hyperparameter 
	learning_rate = 0.0001
	#y = mx + b : slope function 
	initial_b = 0
	initial_m = 0
	num_iteration = 1000
	[b,m] = gradient_descent_runner(points, initial_m,initial_b,learning_rate,num_iteration)
	print(b)
	print(m)





if __name__=='__main__':
	run()
