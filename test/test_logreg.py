"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

def test_updates():
	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	# What is a reasonable loss?
	
	#Positive Test Case: Gradient 
    num_feats = 3  # Number of features in the input # Setted up a simple instance of LogisticRegression with known weights.
    model = LogisticRegression(num_feats) # Initialize the model and store it in a variable.
    model.weights = np.array([0.1, 0.2, -0.1])  # Example weights for testing
    X = np.array([[1.0, 2.0, 3.0], # Define feature matrix X (2 samples, 3 features)
                  [4.0, 5.0, 6.0]]) 
    y = np.array([1, 0]) # Define labels y (2 samples)
    expected_gradient = np.array([0.02, -0.03, 0.04])  # Manually calculated expected gradient
    calculated_gradient = model.calculate_gradient(X, y) # Calculated gradient using model.
    assert np.allclose(calculated_gradient, expected_gradient, atol=1e-5), \
    	"Positive test failed: Calculated gradient does not match the expected value." # Checked that the calculated gradient is close to the expected gradient.
 

    # Negative Test Case: Gradient
    X_zero = np.zeros((2, 3)) # Defined an input where X has all zero values. 
    y_zero_test = np.array([1, 0])  # Defined labels as before.  
    gradient_zero = model.calculate_gradient(X_zero, y_zero_test) # Calculated gradient with zero features
    assert np.allclose(gradient_zero, np.zeros_like(gradient_zero)), \
        "Negative test failed: Gradient with zero features should be zero."

    # Set up a simple instance of LogisticRegression
    num_feats = 3  # Number of features in the input
    model = LogisticRegression(num_feats)
    
    # Positive Test Case: Loss function 
    model.weights = np.array([0.1, -0.2, 0.3])  # Example weights for testing
    X = np.array([[1.0, 0.5, -1.5], # Defined the feature matrix X. 
                  [-1.0, 2.0, 0.3]])  # shape (2, 3)
    y = np.array([1, 0])  # Defined labels y. 
    expected_loss = 0.693  # Binary cross-entropy loss. 
    calculated_loss = model.loss_function(X, y) # Calculated loss using the model. 

    # Checking that the calculated loss is close to the expected loss. 
    assert np.isclose(calculated_loss, expected_loss, atol=1e-3), \
        "Positive test failed: Calculated loss does not match the expected value."

    # Negative Test Case: Loss Function 
    model.weights = np.array([1000, 1000, 1000])  # Extreme weights, high positive value to force all predictions close to 1. 
    extreme_loss = model.loss_function(X, y) # Calculated loss, which should be high. 
    
    # Loss should be large in this case. 
    assert extreme_loss > 10, \
        "Negative test failed: Loss should be high for extreme weights leading to poor predictions."
    

def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?

	# Check accuracy of model after training

	pass