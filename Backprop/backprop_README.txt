CSC446 Machine Learning - HW 3 - Jingwen Zhong

What (combinations of) hyperparameter values did you try and why? 
	I tried to decide the suibale number of iterations.
	I'm still working on change the hidden dimentions and learning rate too.

What criteria did you use for selecting the best ones?
	I used Cross-entropy loss and accurcy to decide.

	From what I've learned on the internet about the the dev set part:
		Through the accuracy of the model on the dev set, 
		the ratio of the number of samples correctly classified by the model to the total number of samples 
		is calculated to measure the effect of the model. The goal is to measure the effect of the model.

	T	o update our model parameters, we can calculation of the loss function, 
		the goal is to reduce the optimization error (Optimization error), 
		to reduce the empirical risk of the model under the combined action 
		of the loss function and the optimization algorithm.

Please also include the results from all of those experiments.
	see pic Zhong_backprop_loss.npg and pic Zhong_backprop_accu.npg

Discuss the results: Do they make sense?
	The result makes sense. The loss decreases very slowly(almost fat) and the accuracy increases slowly
		for both dev and training set after point 4 .
	I believe for all the machine to learn, after certain point the speed of performance increase 
		and increase of accuracy will become more and more slow, but it still can continue to improve
	
Do they suggest any conclusions about the importance of the different hyperparameters to the performance of the algorithm? 
	Yes, as I wrote before after certain point, the speed of performance increase 
		and increase of accuracy will become more and more slow, but it still can improve by for example 
		selecting hyperameters to make the algorithm better fitted to the training set.


Which combinations of hyperparameter values worked best?
	iteration 4