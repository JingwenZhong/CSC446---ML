CSC446 Machine Learning - HW 7 - Jingwen Zhong

1. Init_model:

	Since EM algorithm is very senstive to the initial parameters and it is possible to get very different results, so I took a 
look at the data. (still don't know how to choose the best initial parameters)
	For each mu: Generate a random number from normal distribution with mean=1, sigma = 2 since the mus for the training data are 
around this range.
	For each lambda: Generate a random number from uniform distribution form 0 to 1 to control the range of lambda which is (0,1)
	For sigmas: each sigma looks like: array([[1., 0.],[0., 1.]]), 
		    If it's tied covariance then it will look like: array([[1., 0.],[0., 1.]])


2. Hyperparameter(seperate covaruabce setting):

	a. cluster_num = 3, iteration = 50. (Zhong_em_gaussian_ll_1.jpg)

		I plotted the training data, and I thought the data can be classified into 3 cluster, and it might be the best choice. 
	So for the hyperparameter first I tried cluseter_num = 3.
		From that graph, we can see that the results do not change that much after round iteration 20. 
		The final result of log likelihood for training data is around -4.41, for dev data is around -4.43.
		To comfirm my thought, I tried iteration = 50(Zhong_em_gaussian_ll_2.jpg), the final result is the almost the same,
	and the line does become flat after iteration 20.
		I also tried iteration 30 and 40(Zhong_em_gaussian_ll_3.jpg), both of their final result of log likelihood for 
	training data is around -4.5, and for dev data is around -4.45. And from this plot we can clearly see that 
	results do not change that much after round iteration 20. 
		Although from the plot it does not change much after iteration 20,  the final result of iteration 50 is better. 
	I still choose iteration 50 when the cluster_number =3.


	b. cluster_num = 5, iteration = 50. (Zhong_em_gaussian_ll_4.jpg)

		Since I decided the iteration to be 50, I tried cluster number 5. The final result of log likelihood for 
	training data is around -4.35, and for dev data is around -4.325.
		The result is better than the cluster number 3, and it is unknown why the dev data perform than training data.
	It might because the training data has most hard cases and dev data has easier cases.
		Then I tried cluster number 7, and iteration 50. The final result of log likelihood is similar to cluster number 5.

	c. cluster_num = 10, iteration = 50. (Zhong_em_gaussian_ll_5.jpg)
		Just to be curious, I plot cluster_num =10 and iteration 50, and the graph shows that it starts to be overfitting after
	around iteration 20.

		
	Therefore the best combination of hyperparameters I will choose is cluster_num = 5, iteration = 50. 


3. --tied:

	I also plot the the standard covariance with cluster_num = 5 and iteration = 50(Zhong_em_gaussian_ll.jpg). The final result of 
log likelihood for training data is around -4.60, and for dev data is around -4.62, which confirm my assumption that the result of
separate covaraince setting is better than standard covariance setting since the shape for the cluster in standard covariance setting is
the same, and it is not as flexible as the separate covaraince setting.

	Note that since log likelihood for the initial parameter is very low(lower than -7), it is hard to observe later results on the graph,
so I didn't plot that point.