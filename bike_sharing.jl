# # Bike sharing demand forecast (on daily data)

# Predict bike sharing demand as a function of seasonal and weather
# conditions.

# Data origin:
# - original full dataset (by hour, not used here): https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
# - simplified dataset (by day, with some simple scaling): https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec
#    - description: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf
#    - data: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip

# Note that even if we are estimating a time serie, we are not using
# here a recurrent neural network as we assume the temporal dependence
# to be negligible (i.e. Y_t = f(X_t) alone).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import Random
Random.seed!(123);

# Loading cleaned version of the data:

using UrlDownload, ZipFile, PooledArrays, DataFrames, CSV
location = "https://www.hds.utc.fr/~tdenoeux/"*
    "dokuwiki/_media/en/bike_sharing_day.csv.zip"
data = DataFrame(urldownload(location));
describe(data)

# The target `y` is the `:cnt` column. The features, `x`, are to include everything that's left, excluding `:dteday`, `:casual`, and `:registered`:

using MLJ
y, x = unpack(data,
              ==(:cnt),
              name -> !(name in [:dteday, :casual, :registered]));

# Inspecting the interpretation of the data as currently represented:

schema(x)

#-

scitype(y)

# Coercing type of `y` to get appropriate interpretation:

y = coerce(y, Continuous);

# Optional: look for all supervised models that apply to this data:

models(matching(x, y))

# Splitting the data into train, validation and test sets:

train, val, test = partition(eachindex(y), 0.75, 0.125);
xtrain    = x[train, :]
ytrain    = y[train]
xval      = x[val, :]
yval      = y[val]
xtest     = x[test, :]
ytest     = y[test];

# ## Decision Trees

# Loading the model code (a default instance is returned):

tree_model = @load DecisionTreeRegressor pkg=DecisionTree

# Note that a model is just a struct containing hyperparameters. The
# learned parameters (in this case a tree) called `tree` below.


# ### Tuning `max_depth` (by hand, using a holdout set - no cross validation)

function findBestDepth(xtrain, ytrain, xval, yval, attemptedDepths)
    best_depth = 1
    best_error   = +Inf
    for ad in attemptedDepths
        tree_model.max_depth = ad
        tree, _ = MLJ.fit(tree_model, 0, xtrain, ytrain) # verbosity=0
        ŷval   = MLJ.predict(tree_model, tree, xval)
        mean_proportional_error = mape(ŷval, yval)
        println("$ad : $mean_proportional_error")
        if mean_proportional_error < best_error
            best_depth = ad
            best_error   = mean_proportional_error
        end
    end
    return (best_depth, best_error)
end

#-

best_depth, best_error = findBestDepth(xtrain, ytrain, xval, yval, 1:20)

# Re-training the best model on the train set:

model = DecisionTreeRegressor(max_depth=best_depth)
tree, _ = MLJ.fit(model, 1, xtrain, ytrain)

#-

ŷtrain = MLJ.predict(model, tree, xtrain)
ŷval   = MLJ.predict(model, tree, xval)
ŷtest  = MLJ.predict(model, tree, xtest);

#-

@show mape(ŷtrain, ytrain) mape(ŷval, yval) mape(ŷtest, ytest)

#-

using StatsPlots
pyplot()

#-

scatter(ytrain,
        ŷtrain,
        xlabel="daily rides",
        ylabel="est. daily rides",
        label=nothing,
        title="Est vs. obs in training period")

#-

scatter(yval,ŷval,
        xlabel="daily rides",
        ylabel="est. daily rides",
        label=nothing,
        title="Est vs. obs in validation period")

#-

scatter(ytest,
        ŷtest,
        xlabel="daily rides",
        ylabel="est. daily ri des",
        label=nothing,
        title="Est vs. obs in testing period")

using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=true) #src
Literate.notebook(@__FILE__, @__DIR__, execute=true) #src






