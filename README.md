# Practical_Application_Assignment_11.1
Repo contents:
1. [ichinose-usedcar.ipynb](ichinose-usedcar.ipynb) - jupyter notebook
2. [vehicles.csv](vehicles.csv) - used car dataset in CSV format
Machine Learning Approach to Used Car Pricing Modeling

Business Understanding

The objective of this study is to determine what key factors drives the price of a used car. The business requirements are from the client who owns a used car dealership which requires information about what consumers value in a used car and information useful for pricing trade-ins as an alternative to kbb.com. The goal is to develop a set of factors that drive up or down a value of a car in dollar values and a model useful for predicting the price of a vehicle based on these factors.

The plan involves:

1. Data understanding
    A. An exploration and inventory of the dataset 
    B. Identify which features are continuous and numerical versus which are discrete and categorical
    C. Identify the features and target for regression

2. Data preparation
    A. Window the appropriate ranges of the features and targets
    B. Removal of the data with missing key features
    C. Removal of data outliers as a form of data quality control 
    D. Removal of duplicates 
    E. Logarithmic and exponential transformations of the target to achieve normal distributions
    F. Encoding categorical data (ordinal or one hot encoding).

3. Modeling
    A. We will compare RIDGE and LASSO regression
    B. Set up a pipeline workflow to handle the transformations
    C. Set up a grid search to determine the best regularization hyperparameter alpha
    D. The regularization approach allows us to include many features without overfitting.
    E. Examination of the residuals over the target range
    F. Cross validation to check for overfitting

4. Evaluation
    A. Analysis of Resulting Coefficients
    B. Importance of Coefficients
    C. Interpretation of Coefficients
    D. Implications to used car business
    E. Future directions

5. Deployment
    A. Direct application: Comparison of the model to kbb.com with two example cars 

Key findings:
1. Value of vehicles decrease with age and higher odometer readings. One would also expect inverse relation between years and odometer. 10 years is the age where car values flatten out < 10k. This is also where the odometer also flattens out around ~110k miles.
2. The model indicates so logical expectations that vehicles that are specialized (e.g., offroad, truck) or have more power rank higher with positive coefficients leading to higher value while economy vehicles with lower power rank lower leading to decrease in vehicles resale value.
3. Year, cylinders, and odometer were the most important.
4. We identify some variability in coefficients that may not be well resolved. These include categories (fuel_other, type_other, cylinder_other) that involve other selections that may not be well represented in the dataset. Also, less represented features like 3-cylinders, offroad, bus, fuel_electric may not be well represented.
5. We demonstrate that for two example vehicles (2013 Honda Fit and 2015 Toyota Tacoma) that the model predicts the value of the vehicle within the range of the Kelly Blue Book value (kbb.com). This demonstrates the usefulness of the model.

What does this mean for implications in fine tuning inventory?
1. Age and year play an important role. Values of cars flatten out after 10 years or 110k miles. Dealers may want to adjust inventory to maximize profits accordingly.
2. Consumers want fun (offroad, trucks, 4wd) and powerful (10,8,6 cylinder) vehicles. Vehicles with these features are valued more. Dealers may want to stock these kinds of cars as opposed to vehicles that have negative coefficients. These tended to be more economy or budget cars with lower power.

Next steps, future directions, and recommendations:
1. Examine the residuals group by the following to check for trends:
  A. region
  B. model and make
  C. color
  These features were not used in the regression. If trends exist, then future work could include adding these.

2. Revisit dataset in a few years to see if there is increase in value for fuel efficient vehicles due to gas inflation.
3. Understand why the model has more misfit for newer used cars (1 to 2 years old). Improving prediction can help pricing newer used cars which will have more value.
