# Practical_Application_Assignment_11.1

Repo contents:
1. [ichinose-usedcar.ipynb](ichinose-usedcar.ipynb) - jupyter notebook

# Machine Learning Approach to Used Car Pricing Modeling

## Business Understanding

The objective of this study is to determine what key factors drives the price of a used car. The business requirements are from the client who owns a used car dealership. They require information about what consumers value in a used car and information useful for pricing inventory and trade-ins as an alternative to companies like kbb.com. The goal is to develop a set of factors that drive up or down a value of a car in dollar values and a model useful for predicting the price of a vehicle based on these factors.

The plan involves:

<ol type="1">
    <li><a href="#dataunderstanding">Data understanding</a></li>
    <ol type="A">
        <li>An inventory of the dataset</li>
        <li>Identify which features are continuous and numerical versus which are discrete and categorical</li>
        <li>Explore the unique range of categorical data</li>
    </ol>

<li><a href="#datapreparation">Data Preparation</a></li>
    <ol type="A">
        <li>Window the appropriate ranges of the features and targets</li>
        <li>Removal of the data with missing key features</li>
        <li>Removal of data outliers as a form of data quality control</li> 
        <li>Removal of duplicates</li> 
        <li>Logarithmic and exponential transformations of the target to achieve normal distributions</li>
        <li>Identify the features and target for regression</li>
        <li>Split dataset for training and teseting</li>
    </ol>
    
<li><a href="#modeling">Modeling</a></li>
    <ol type="A">
        <li>Set up a pipeline workflow to handle the transformations and linear regressions</li>
        <li>Encoding categorical data using ordinal and one hot encoding</li>
        <li>We will compare RIDGE and LASSO linear regression models</li>
        <li>Set up a grid search to determine the best regularization hyperparameter alpha. The regularization approach allows us to include many features without overfitting.</li>
        <li>Examination of the observed and predicted target "independent" variables</li>
        <li>Cross validation to check for overfitting</li>
    </ol>
<li><a href="#evaluation">Evaluation</a></li>
    <ol type="A">
        <li><a href="#coefficients">Analysis of Resulting Coefficients</a></li>
        <li><a href="#importance">Importance of Coefficients</a></li>
        <li><a href="#transform">Transform Target Regressor Test</a></li>
        <li><a href="#variability">Coefficient Importance and Variability Analysis</a></li>
        <li><a href="#residualanalysis">Residual Analysis</a></li>
    </ol>
        
 <li><a href="#deployment">Deployment</a></li>
    <ol type="A">
        <li><a href="#direct">Direct application: Comparison of the regression model to kbb.com</li> 
        <li><a href="#keyfindings">Key findings</a></li>
        <li><a href="#implications">Business implications</a></li>
        <li><a href="#future">Future directions</a></li>
    </ol>
</ol>

## Key findings:

1. Value of vehicles decrease with increasing age and higher odometer readings. We also observe an inverse relation between vehicle year and odometer. At an age of ten years and older, car values flatten out at less than $10,000. This is also where the odometer readings flatten out at approximately 110,000 miles.

2. The regression model indicates that vehicles which are specialized (e.g., off-road, truck) or have more power (cylinders > 6), rank higher with positive coefficients leading to increase value while economy vehicles with lower power rank lower leading to a decrease in vehicle's resale value. In other words, consumers want fun vehicles with more power. 

3. The car features year, cylinders, and odometer were the most important.

4. We identify some variability in several coefficients that indicate that they are not well resolved. These include 'other' categories (e.g., fuel_other, type_other, cylinder_other) that may not be well represented in the dataset. Also, less represented features like 3-cylinders, off-road, bus, fuel_electric may not be well represented in the dataset.

5. We demonstrate that for two example vehicles (2013 Honda Fit and 2015 Toyota Tacoma) that the model predicts the value of the vehicle within the range of the Kelly Blue Book value (kbb.com). This demonstrates the usefulness of the model.

### What does this mean for implications in fine tuning inventory?

1. Age or year and odometer mileage play an important role in the value of a used car. Values of cars flatten out after 10 years or 110k miles. Dealers may want to adjust inventory to maximize profits accordingly.

2. Consumers want fun, useful (off-road, trucks, four-wheel-drive) and powerful (10,8,6 cylinder) vehicles. Vehicles with these features are valued more. Dealers may want to stock these kinds of cars as opposed to vehicles that have negative coefficients that lower the values of used cars. These tended to be more economy or budget cars with less powerful engines.

### Next steps, future directions, and recommendations:

1. We recommend further examination of the regression residuals group by the following to check for trends:

  A. region

  B. model and make

  C. color

  These features were not used in the regression. If trends exist, then future work could include adding these.

2. Revisit the dataset in a few years to see if there is an increase in value for fuel efficient vehicles due to gas inflation.

3. We would like to further understand why the model has more misfit for newer used cars (1 to 2 years old). As above, it is possible that the Covid-19 pandemic has injected some nonlinearity to the used car market with supply chain issues.  Improving prediction can help pricing newer used cars which will have more value.
