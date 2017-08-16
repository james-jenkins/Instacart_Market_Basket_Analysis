# TODO: overall XGBoost solution
# TODO: separate weak classifiers for different ailes
# TODO: try weighted average based on multiple classifiers
# TODO: keyword analysis?

# DEPENDENCIES ------------------------------------------------------------
library(readr)
library(tidyverse)
library(caret)
library(xgboost)
library(doParallel)
library(MLmetrics)

rm(list = ls())
gc()

# DATA PREPARATION --------------------------------------------------------

# Data Import -------------------------------------------------------------
path <- "../Data Extracted/"

aisles           <- read_csv(file.path(path, "aisles.csv"))
departments      <- read_csv(file.path(path, "departments.csv"))
order_prior      <- read_csv(file.path(path, "order_products__prior.csv"))
order_train      <- read_csv(file.path(path, "order_products__train.csv"))
orders           <- read_csv(file.path(path, "orders.csv"))
products         <- read_csv(file.path(path, "products.csv"))


# Data wrangling ----------------------------------------------------------

aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)
products$product_name <- as.factor(products$product_name)

products                      %>%
  inner_join(aisles)          %>%
  inner_join(departments)     %>%
  mutate(aisle_id = NULL, 
         department_id = NULL) ->
  products_location

order_train$user_id <- orders$user_id[match(order_train$order_id, orders$order_id)] 

orders_content <- inner_join(orders, order_prior, by = "order_id")

rm(products, order_prior, aisles, departments)
gc()

# FEATURE CONSTRUCTION ----------------------------------------------------


# Product Features --------------------------------------------------------

orders_content                        %>%
  group_by(product_id)                %>%
  summarise(product_order_count          = n(),
                   product_reorder_count        = sum(reordered > 0, na.rm = TRUE),
                   product_reorder_proportion   = product_reorder_count/product_order_count,
                   product_first_order_count    = sum(reordered == 0, na.rm = TRUE)) %>%
  inner_join(products_location, by = "product_id") -> 
  product_predictors

total_orders <- sum(as.numeric(product_predictors$product_order_count))

orders_content                                             %>%
  inner_join(products_location, by = "product_id")         %>%
  group_by(aisle)                                          %>%
  summarise(aisle_order_count     = n(),
         aisle_popularity      = aisle_order_count/total_orders,
         aisle_reorder_count   = sum(reordered > 0, na.rm = TRUE),
         aisle_unique_products = length(unique(product_id))) ->
  aisle_predictors

product_predictors <- inner_join(product_predictors, aisle_predictors, by = "aisle")
rm(aisle_predictors)
gc()

orders_content %>%
  inner_join(products_location, by = "product_id") %>%
  group_by(department) %>%
  summarise(dep_order_count     = n(),
         dep_popularity      = dep_order_count/total_orders,
         dep_reorder_count   = sum(reordered > 0, na.rm = TRUE),
         dep_unique_products = length(unique(product_id))) ->
  dep_predictors

product_predictors <- inner_join(product_predictors, dep_predictors, by = "department")
rm(dep_predictors, products_location)
gc()

product_predictors <- select(product_predictors, -product_name, -aisle, -department)

# User Features -----------------------------------------------------------

orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(user_orders  = max(order_number),
            user_avg_lag = mean(days_since_prior_order, na.rm = TRUE)) ->
  user_orders

orders_content %>%
  group_by(user_id) %>%
  summarise(user_total_products = n(),
            user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
            user_unique_products = length(unique(product_id))) ->
  user_products

user_orders <- inner_join(user_orders, user_products)

orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, 
         order_id, 
         eval_set,
         days_since_prior_order) ->
  user_test

user_orders <- inner_join(user_orders, user_test)

rm(user_products)
gc()

# TRAINING & TESTING SETS -------------------------------------------------

orders_content %>%
  group_by(user_id, product_id) %>%
  summarise(current_orders = n(),
            current_first_order = min(order_number), 
            current_last_order = max(order_number)) %>%
  inner_join(product_predictors, by = "product_id") %>%
  inner_join(user_orders, by = "user_id")->
  df

df <- left_join(df, order_train %>% select(user_id, product_id, reordered),
                  by = c("user_id", "product_id"))

df <- data.frame(df)

train <- df                   %>% 
  filter(eval_set == "train") %>% 
  select(-eval_set, -user_id, -product_id, -order_id)

train$reordered[is.na(train$reordered)] <- 0

test  <- df                  %>% 
  filter(eval_set == "test") %>% 
  select(-eval_set, -user_id, -reordered)

rm(df, orders_content, product_predictors)
gc()

# MODEL -------------------------------------------------------------------
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  c(F1 = f1_val)
}

xgb_grid <- expand.grid(nrounds   = 150,
                        eta       = c(1, 0.1, 0.01, 0.001, 0.0001),
                        max_depth = c(2, 4, 6, 8, 10),
                        gamma     = c(0.5, 0.75, 0.1),
                        colsample_bytree = c(0.6, 0.8),
                        min_child_weight = c(2,4,6),
                        subsample = seq(0.5, 1, length = 5))
                                                                                            
xgb_trControl <- trainControl(method          = "cv",
                              number          = 5,
                              # repeats         = 10,
                              verboseIter     = TRUE,
                              # returnData      = FALSE,
                              # returnResamp    = "all",
                              classProbs      = TRUE,
                              summaryFunction = f1,
                              allowParallel   = TRUE )

if(file.exists("C:/Users/james.jenkins/Desktop/output.txt"))
  file.remove("C:/Users/james.jenkins/Desktop/output.txt")

cl <- makePSOCKcluster(3, outfile = "C:/Users/james.jenkins/Desktop/output.txt")
registerDoParallel(cl)
# train <- train %>% sample_frac(0.01)
model_xgbLinear <- train(x        = as.matrix(train %>% select(-reordered)),
                       y          = as.factor(make.names(train$reordered)),
                       trControl  = xgb_trControl,
                       metric     = "F1",
                       tuneLength = 5,
                       method     = "xgbLinear",
                       verbose    = TRUE)

# model_xgbTree <- train(x         = as.matrix(train %>% select(-reordered)),
#                          y         = as.factor(make.names(train$reordered)),
#                          trControl = xgb_trControl,
#                          tuneGrid  = xgb_grid,
#                          method    = "xgbTree",
#                          verbose   = TRUE)
stopCluster(cl)
# 
# save(model_xgbTree, file = "model_tree.rds")
save(model_xgbLinear, file = "model_linear.rds")

# load("model.rds")

nrounds = 150
lambda  = 0.1
alpha   = 0.1
eta     = 0.3

# SUBMISSION --------------------------------------------------------------

test$reordered <- predict(model_xgbLinear, test %>% select(-order_id, -product_id))

submission <- test %>%
  filter(reordered == "X1") %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)

