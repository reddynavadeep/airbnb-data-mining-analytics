library(tidyverse)
library(text2vec)
library(tm)
library(SnowballC)
library(glmnet)
library(vip)
library(naivebayes)
library(ranger)
library(xgboost)
library(ROCR)
library(rsample)
library(caret)
library(randomForest)
library(fastDummies)
library(Matrix)
library(text2vec)
library(lightgbm)

set.seed(12345)

# Reading csv files
train_x <- read_csv("airbnb_train_x_2023.csv")
train_y <- read_csv("airbnb_train_y_2023.csv")
test_x <- read_csv("airbnb_test_x_2023.csv")
# To get income distribution in the zip codes
income_data_cleaned <- read_csv("income_data_cleaned.csv") 
# To get population and population density in the zip codes
pop_density_clean <- read_csv("population_density.csv") 
# To get popular tourist destinations
popular_cities <- read.csv("Popular_tourist_destination.csv")
# To get Top cities in the USA
popular_cities_yahoo <- read.csv("top_cities.csv")


###########################################################################################
#################################STEP 1: DATA PREPARATION##################################
###########################################################################################
# Combining train and test data and setting "perfect_rating_score", and "high_booking_rate" to numeric values
x_data <- rbind(train_x, test_x)
train_y <- train_y %>%
  mutate(perfect_rating_score = ifelse(perfect_rating_score == "YES", 1, 0),
         high_booking_rate = ifelse(high_booking_rate == "YES", 1, 0))

# Cleaning the zipcode in our data to remove area codes.
x_data <- x_data%>%mutate(
  zipcode = as.character(zipcode),
  zipcode = lapply(strsplit(x_data$zipcode, "-"), `[`, 1),
  zipcode = as.character(zipcode),
  zipcode = lapply(strsplit(x_data$zipcode, "\\."), `[`, 1),
  zipcode = as.numeric(zipcode))

# Changing the names of the columns in pop_density_clean, and income_data_cleaned
colnames(pop_density_clean) <- c("zipcode","population","density")
colnames(income_data_cleaned) <- c("zipcode","income_families","income_non_families")
income_data_cleaned<- income_data_cleaned%>% mutate(zipcode = as.numeric(zipcode))

# Joining data and income, population, and density on the basis of zipcode column
x_data <- left_join(x_data, income_data_cleaned, by = "zipcode")
x_data <- left_join(x_data, pop_density_clean, by = "zipcode")

# Select city variable
popular_cities <- popular_cities %>% select("City")
colnames(popular_cities) <- c("city")
# Create a new factor describing if the city is a popular tourist destination
popular_cities <- popular_cities$city
x_data <- x_data%>% mutate(popular_destination = as.factor(ifelse(city_name %in% popular_cities,1,0)))

# Create a new factor describing if the city is a popular destination from yahoo data
popular_cities_yahoo <- popular_cities_yahoo$city
x_data <- x_data%>% mutate(popular_destination_yahoo = as.factor(ifelse(city_name %in% popular_cities_yahoo,1,0)))

###########################################################################################
##################################STEP 2: DATA CLEANING####################################
###########################################################################################
# Cleaning variables:
# For cancellation_policy: {strict, super_strict_30} --> {strict}
# Convert cleaning_fee and price into numbers
# Replace NAs in cleaning_fee and price with 0
# Replace NAs in beds, bedrooms, host_total_istings_count with their mean

# Creating New Features new features:
# price_per_person is the nightly price per accommodates 
# has_cleaning_fee - {YES, NO}
# bed_category - {bed, other}
# property_category - {apartment, hotel, condo, house, other}
# ppp_ind is 1 if the price_per_person is greater than the median for the property_category, and 0 otherwise

airbnb_clean <- x_data %>%
  group_by(cancellation_policy) %>%
  mutate(cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30"), 'strict', cancellation_policy)) %>%
  ungroup() %>%
  mutate(cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
         cleaning_fee = as.numeric(parse_number(cleaning_fee)),
         price = ifelse(is.na(price), 0, price),
         price = as.numeric(parse_number(price)),
         bedrooms = as.numeric(ifelse(is.na(bedrooms), mean(bedrooms, na.rm=TRUE), bedrooms)),
         beds = as.numeric(ifelse(is.na(beds), mean(beds, na.rm=TRUE), beds)),
         host_total_listings_count = as.numeric(ifelse(is.na(host_total_listings_count), mean(host_total_listings_count, na.rm=TRUE), host_total_listings_count))) %>%
  mutate(price_per_person = as.numeric(round(price/accommodates, 2)),
         has_cleaning_fee = as.factor(ifelse(cleaning_fee == 0, "NO", "YES")),
         bed_category = as.factor(ifelse(bed_type == "Real Bed", "bed", "other")),
         property_category = ifelse(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "apartment", property_type),
         property_category = ifelse(property_category %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel", property_category),
         property_category = ifelse(property_category %in% c("Townhouse", "Condominium"), "condo", property_category),
         property_category = ifelse(property_category %in% c("Bungalow", "House"), "house", property_category),
         property_category = ifelse(property_category %in% c("apartment", "hotel", "condo", "house"), property_category, "other"),
         property_category = as.factor(property_category),
         property_type = as.factor(property_type)
  ) %>%
  group_by(property_category) %>%
  mutate(ppp_ind = as.numeric(ifelse(price_per_person > median(price_per_person, na.rm=TRUE), 1, 0))) %>%
  ungroup() %>%
  mutate(bed_type = as.factor(bed_type),
         cancellation_policy = as.factor(cancellation_policy),
         room_type = as.factor(room_type),
         ppp_ind = as.factor(ppp_ind))


########################################################################################################
# Cleaning variables:
# Replace NAs in bathrooms with the median value
# Replace NAs in host_is_superhost with FALSE

# Creating New Features new features:
# "charges_for_extra" - {YES, NO}
# "host_acceptance" - {ALL, SOME, MISSING}
# "host_response"  - {ALL, SOME, MISSING}
# "has_min_nights" - {YES, NO}
# Replace market with "OTHER" if there are under 300 instances in that market. Convert market to a factor.
airbnb_clean <- airbnb_clean %>%
  mutate(bathrooms = as.numeric(ifelse(is.na(bathrooms), median(bathrooms, na.rm=TRUE), bathrooms)),
         host_is_superhost = ifelse(is.na(host_is_superhost), FALSE, host_is_superhost),
         extra_people = as.numeric(parse_number(extra_people)),
         charges_for_extra = as.factor(ifelse(extra_people > 0, "YES", "NO")),
         host_acceptance = as.factor(case_when(is.na(host_acceptance_rate) ~ "MISSING",
                                               host_acceptance_rate == 1.00 | host_acceptance_rate == "100%"~ "ALL",
                                               TRUE ~ "SOME")),
         host_response = as.factor(case_when(is.na(host_response_rate) ~ "MISSING",
                                             host_response_rate == 1.00 |  host_response_rate == "100%" ~ "ALL",
                                             TRUE ~ "SOME")),
         has_min_nights = as.factor(ifelse(minimum_nights > 1, "YES", "NO"))) %>%
  group_by(market) %>%
  mutate(market = as.factor(ifelse(n() < 300, "OTHER", market))) %>%
  ungroup()

########################################################################################################
# Cleaning host_since, and first_review
airbnb_clean<-airbnb_clean%>%
  mutate(list_till_today=Sys.Date()-host_since,
         last_review_till_today=Sys.Date()-first_review,
         list_till_today = as.numeric(list_till_today),
         last_review_till_today = as.numeric(last_review_till_today))

# Cleaning interaction, access, description, house_rules, neighborhood_overview, and market
# We would be assigning NAs as missing in these fields
airbnb_clean <- airbnb_clean %>% 
  mutate(list_till_today =  ifelse(is.na(list_till_today), mean(list_till_today, na.rm=TRUE), list_till_today),
         interaction = ifelse(is.na(interaction), "MISSING", interaction),
         access = ifelse(is.na(access), "MISSING", access),
         description = ifelse(is.na(description), "MISSING", description),
         house_rules = ifelse(is.na(house_rules), "MISSING", house_rules),
         neighborhood_overview = ifelse(is.na(neighborhood_overview), "MISSING", neighborhood_overview),
         market = ifelse(is.na(market), "MISSING", market))

# Cleaning population, density, income_families, income_non_families, zipcode
# We would be assigning mean to the NA values, and marking 0 in the NA values
airbnb_clean <- airbnb_clean %>% 
  mutate(population =  ifelse(is.na(population), mean(population, na.rm=TRUE), population),
         density =  ifelse(is.na(density), mean(density, na.rm=TRUE), density),
         income_families = ifelse(is.na(income_families), mean(income_families, na.rm=TRUE), income_families),
         income_non_families = ifelse(is.na(income_non_families), mean(income_non_families, na.rm=TRUE), income_non_families),
         zipcode =  ifelse(is.na(zipcode), 0, zipcode))

# Cleaning monthly_price, and weekly_price to use as a factor
# If they are present, we mark it as 1, otherwise 0
# All security deposit fields that are NAs are marked as 0
# is_business_travel_ready is marked as FALSE, if they are NAs
# All licenses that are NAs are marked as False, and all the others are marked as TRUE
airbnb_clean <- airbnb_clean%>%mutate( monthly_price = ifelse(is.na(monthly_price), 0, monthly_price),
                                       monthly_price = as.numeric(parse_number(monthly_price)),
                                       monthly_price = as.factor(ifelse(monthly_price==0, 0, 1)),
                                       weekly_price = ifelse(is.na(weekly_price), 0, weekly_price),
                                       weekly_price = as.numeric(parse_number(weekly_price)),
                                       weekly_price = as.factor(ifelse(weekly_price==0, 0, 1)),
                                       security_deposit = ifelse(is.na(security_deposit), 0, security_deposit),
                                       security_deposit = as.numeric(parse_number(security_deposit)),
                                       is_business_travel_ready = ifelse(is.na(is_business_travel_ready), FALSE,is_business_travel_ready),
                                       license = ifelse(is.na(license), FALSE,TRUE))

# Adding popular_destination and popular_destination_yahoo to our dataset
airbnb_clean$popular_destination = x_data$popular_destination
airbnb_clean$popular_destination_yahoo = x_data$popular_destination_yahoo
###########################################################################################
###############################STEP 3: FEATURES FROM TEXT##################################
###########################################################################################
# Creating word embeddings from Interaction, description, and access. Using them to create a weighted
# matrix for each observation in original data. It would be used to assign centres to our data.
# This is done in order to convey the meaning of the text on a scale.
########################################## INTERACTION ###########################################
set.seed(12345)
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    word_tokenizer 
}
# Creating i-token
it_interaction = itoken(airbnb_clean$interaction, 
                  preprocessor = tolower, 
                  tokenizer = cleaning_tokenizer, 
                  progressbar = FALSE)
# Creating vocabulary
vocab_interaction <- create_vocabulary(it_interaction, ngram = c(1,3))
# Pruning Vocabulary
pruned_vocab_interaction <- prune_vocabulary(vocab_interaction, term_count_min = 5)
# Making a dtm and tcm (for getting the word embeddings)
vectorizer_interaction = vocab_vectorizer(pruned_vocab_interaction)
dtm_interaction <- create_dtm(it_interaction, vectorizer_interaction)
tcm_interaction <- create_tcm(it_interaction, vectorizer_interaction,  skip_grams_window = 5L)

set.seed(12345)
# making a glove and mark the number of embeddings for each word
glove = GlobalVectors$new(rank = 50, x_max = 10)
# fititng this to calculate our word embeddings
model_interaction = glove$fit_transform(tcm_interaction, n_iter = 10, 
                                        convergence_tol = 0.01, n_threads = 8)
# Creating a matrix with word_vector_interactions, which is square matrix with embeddings related to each word
word_vectors_interactions = model_interaction + t(glove$components)
# Creating a weigthed dtm, with each rows representing number of observations, and
# columns representing number of embeddings that we chose for our glove
weighted_dtm <- dtm_interaction %*% word_vectors_interactions
set.seed(12345)
# Assigning centres to these embeddings using kmeans clustering
kmeans_result <- kmeans(weighted_dtm, centers = 200, iter.max = 150)
# Adding these centres to our dataset
airbnb_clean$cluster_interaction <- kmeans_result$cluster



############################################## ACCESS ##############################################
set.seed(12345)
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    word_tokenizer 
}
# Creating i-token
it_access = itoken(airbnb_clean$access, 
                   preprocessor = tolower, 
                   tokenizer = cleaning_tokenizer, 
                   progressbar = FALSE)
# Creating vocabulary
vocab_access <- create_vocabulary(it_access, ngram = c(1,3))
# Pruning Vocabulary
pruned_vocab_access <- prune_vocabulary(vocab_access, term_count_min = 5)
# Making a dtm and tcm (for getting the word embeddings)
vectorizer_access = vocab_vectorizer(pruned_vocab_access)
dtm_access <- create_dtm(it_access, vectorizer_access)
tcm_access <- create_tcm(it_access, vectorizer_access,  skip_grams_window = 5L)

set.seed(12345)
# making a glove and mark the number of embeddings for each word
glove = GlobalVectors$new(rank = 100, x_max = 10)
# fititng this to calculate our word embeddings
model_access = glove$fit_transform(tcm_access, n_iter = 10, 
                                   convergence_tol = 0.01, n_threads = 8)
# Creating a matrix with word_vector_interactions, which is square matrix with embeddings related to each word
word_vectors_access = model_access + t(glove$components)
# Creating a weigthed dtm, with each rows representing number of observations, and
# columns representing number of embeddings that we chose for our glove
weighted_dtm <- dtm_access %*% word_vectors_access
set.seed(12345)
# Assigning centres to these embeddings using kmeans clustering
kmeans_result <- kmeans(weighted_dtm, centers = 160, iter.max = 200)
# Adding these centres to our dataset
airbnb_clean$cluster_access <- kmeans_result$cluster


############################################## DESCRIPTION ##############################################
set.seed(12345)
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(stopwords()) %>% #remove stopwords
    word_tokenizer 
}
# Creating i-token
it_descrip = itoken(airbnb_clean$description, 
                    preprocessor = tolower, 
                    tokenizer = cleaning_tokenizer, 
                    progressbar = FALSE)
# Creating vocabulary
vocab_descrip <- create_vocabulary(it_descrip, ngram = c(1,3))
# Pruning Vocabulary
pruned_vocab_descrip <- prune_vocabulary(vocab_descrip, term_count_min = 5)
# Making a dtm and tcm (for getting the word embeddings)
vectorizer_descrip = vocab_vectorizer(pruned_vocab_descrip)
dtm_descrip <- create_dtm(it_descrip, vectorizer_descrip)
tcm_descrip <- create_tcm(it_descrip, vectorizer_descrip,  skip_grams_window = 5L)
set.seed(12345)
# making a glove and mark the number of embeddings for each word
glove = GlobalVectors$new(rank = 100, x_max = 10)
# fititng this to calculate our word embeddings
model_descrip = glove$fit_transform(tcm_descrip, n_iter = 10, 
                                    convergence_tol = 0.01, n_threads = 8)
# Creating a matrix with word_vector_interactions, which is square matrix with embeddings related to each word
word_vectors_descrip = model_descrip + t(glove$components)
# Creating a weigthed dtm, with each rows representing number of observations, and
# columns representing number of embeddings that we chose for our glove
weighted_dtm <- dtm_descrip %*% word_vectors_descrip
set.seed(12345)
# Assigning centres to these embeddings using kmeans clustering
kmeans_result <- kmeans(weighted_dtm, centers = 500, iter.max = 150)
# Adding these centres to our dataset
airbnb_clean$cluster_descrip <- kmeans_result$cluster



############################################## AMENITIES ##############################################
set.seed(12345)
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>%
    stemDocument %>%
    word_tokenizer 
}
# Creating i-token
it_amenities = itoken(airbnb_clean$amenities, 
                      preprocessor = tolower, 
                      tokenizer = cleaning_tokenizer, 
                      progressbar = FALSE)
# Creating vocabulary
vocab_amenities <- create_vocabulary(it_amenities)
# Pruning Vocabulary
pruned_vocab_amenities <- prune_vocabulary(vocab_amenities, term_count_min = 500)
# Making a dtm
vectorizer_amenities = vocab_vectorizer(pruned_vocab_amenities)
dtm_amenities <- create_dtm(it_amenities, vectorizer_amenities)

############################################## HOUSE RULES ##############################################

cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>%
    stemDocument %>%
    word_tokenizer 
}
# Creating i-token
it_rules = itoken(airbnb_clean$house_rules, 
                      preprocessor = tolower, 
                      tokenizer = cleaning_tokenizer, 
                      progressbar = FALSE)
# Creating vocabulary
vocab_rules <- create_vocabulary(it_rules)
# Pruning Vocabulary
pruned_vocab_rules <- prune_vocabulary(vocab_rules, term_count_min = 850)
# Making a dtm
vectorizer_rules = vocab_vectorizer(pruned_vocab_rules)
dtm_rules <- create_dtm(it_rules, vectorizer_rules)


#################CREATING A SPARSE MATRIX FROM THE DATAFRAME#################
# Selecting the required cleaned variables from our dataset
set.seed(12345)
perfect_data <- airbnb_clean %>%
  select(accommodates, bedrooms, cancellation_policy, has_cleaning_fee, host_total_listings_count, 
         price, ppp_ind, property_category, bathrooms, charges_for_extra,
         host_acceptance, has_min_nights, market, host_is_superhost, room_type,
         list_till_today, last_review_till_today, cluster_interaction, cluster_access, cluster_descrip,
         population, availability_30, availability_60, availability_90, zipcode, 
         monthly_price, weekly_price, security_deposit, guests_included, instant_bookable,
         is_business_travel_ready, minimum_nights, room_type, is_location_exact, license,
         income_families, income_non_families, density, popular_destination, popular_destination_yahoo)

# Featurizing or making dummies out of these variables
dummy <- dummyVars( ~ . , data=perfect_data, fullRank = TRUE)
perfect_data<- data.frame(predict(dummy, newdata = perfect_data))
set.seed(12345)
# Converting the dataframe to a sparse matrix to use it with word dtms
my_sparse <- sparseMatrix(i = as.integer(row(perfect_data)),
                          j = as.integer(col(perfect_data)),
                          x = as.numeric(unlist(perfect_data)),
                          dims = c(nrow(perfect_data), ncol(perfect_data)),
                          dimnames = list(NULL, names(perfect_data)))
# Combining our sparse matrices with the dtms we created for amenities and rules
my_sparse <- cbind(my_sparse, dtm_amenities, dtm_rules)
###########################################################################################
##################################STEP 4: SEPARATING TEST##################################
###########################################################################################
# Separating data back to the train and test
airbnb_train <- my_sparse[1:dim(train_x)[1], ]
airbnb_test <- my_sparse[(dim(train_x)[1] + 1):(dim(train_x)[1] + dim(test_x)[1]), ]

###########################################################################################
##################################STEP 5: HYPERPARAMETER TUNING############################
###########################################################################################
set.seed(12345)
### Training hyperparameters on a subset of 20000 observations. We do this because the TPR increases 
### very slightly if we include more than 20000 observations. This is supported by our learning curve.
### Our assumption is that the change in TPR will be same even when we include more observations. 
sampled_nrows <- sample(nrow(airbnb_train), 20000, replace = FALSE)
# Sampled dependent and independent variables
sampled_airbnb_train <- airbnb_train[sampled_nrows, ]
sampled_train_y <- train_y[sampled_nrows, ] 


# Sampling the airbnb_train to create train and valid data to tune our hyperparameters
valid_instn = sample(nrow(sampled_airbnb_train), 0.30*nrow(sampled_airbnb_train))
# train_boost_x <- training x data.......valid_boost_x <- validation x data
# train_boost_y <- training dependent data.......valid_boost_y <- validation dependent variables
train_boost_x <- sampled_airbnb_train[-valid_instn, ]
train_boost_y <- sampled_train_y$perfect_rating_score[-valid_instn]
valid_boost_x <- sampled_airbnb_train[valid_instn, ]
valid_boost_y <- sampled_train_y$perfect_rating_score[valid_instn]


# Making the list of hyperparameters that I want to try
depth_list = c(8, 10, 15, 20, 25, 30)
eta_list = c(0.01, 0.05, 0.1, 0.5, 1.0)
nrounds_choose <- c(100, 500, 1000, 1500, 2000, 2500)
max_req_TPR = 0
best_depth = depth_list[1]
best_eta = eta_list[1]
best_nrounds = nrounds_choose[1]


for(i in c(1:length(depth_list))){
  print(i)
  for(j in c(1:length(best_eta))){
    for(k in c(1:length(nrounds_choose))){
      # Training the xgboost model
      xgbst <- xgboost(data = train_boost_x, label = train_boost_y,
                       max.depth = depth_list[i], eta = eta_list[j], nrounds = nrounds_choose[k],
                       objective = "binary:logistic", verbosity = 0, verbose = FALSE)
      # Predicting dependent variable
      preds_bst <- predict(xgbst, valid_boost_x)
      # Making a prediction object to caluclate TPR, and FPR
      pred_full <- prediction(preds_bst, valid_boost_y)
      # Making a ROC object
      roc_full <- performance(pred_full, "tpr", "fpr")
      # Storing TPR, FPR, and cut-off values
      TPR<-roc_full@y.values
      FPR<-roc_full@x.values
      cutoff<-roc_full@alpha.values
      #revise the list to vector
      FPR<-unlist(FPR)
      TPR<-unlist(TPR)
      # Finding the FPR index for FPR nearest to 0.1
      FPR_index<-which.min(abs(FPR-0.1))
      # Storing the TPR for this index
      req_TPR <- TPR[FPR_index]
      # Selecting the hyperparameters for this TPR and storing the hyperparameters
      if(req_TPR > max_req_TPR){
        max_req_TPR = req_TPR
        best_nrounds = nrounds_choose[k]
        best_eta = eta_list[j]
        best_depth = depth_list[i]
      }
    }
  }
}



###########################################################################################
################################STEP 6: MODEL FINALIZATION#################################
###########################################################################################
set.seed(12345)
# Sampling the airbnb_train to create train and valid data
valid_instn = sample(nrow(airbnb_train), 0.30*nrow(airbnb_train))
# train_boost_x <- training x data.......valid_boost_x <- validation x data
# train_boost_y <- training dependent data.......valid_boost_y <- validation dependent variables
train_boost_x <- airbnb_train[-valid_instn, ]
train_boost_y <- train_y$perfect_rating_score[-valid_instn]
valid_boost_x <- airbnb_train[valid_instn, ]
valid_boost_y <- train_y$perfect_rating_score[valid_instn]

set.seed(12345)
# Training the xgboost model with tuned hyperparameters
xgbst <- xgboost(data = train_boost_x, label = train_boost_y, 
               max.depth = 15, eta = 0.05, nrounds = 2000,  
               objective = "binary:logistic")

#################################GENERALIZATION PERFORMANCE#################################
# Predicting dependent variable
preds_bst <- predict(xgbst, valid_boost_x)
# Making a prediction object to caluclate TPR, and FPR
pred_full <- prediction(preds_bst, valid_boost_y)
# Making a ROC object
roc_full <- performance(pred_full, "tpr", "fpr")
# # Plotting the ROC curve for our best model
plot(roc_full, col = "red", lwd = 2)
# Storing TPR, FPR, and cut-off values
TPR<-roc_full@y.values
FPR<-roc_full@x.values
cutoff<-roc_full@alpha.values
summary(FPR)
#revise the list to vector
FPR<-unlist(FPR)
TPR<-unlist(TPR)
cutoff<-unlist(cutoff)
# Finding the FPR index for FPR nearest to 0.1
FPR_index<-which.min(abs(FPR-0.1))
FPR_index
# Choosing a lesser FPR (around 0.0955) to be on the conservative side for our estimations 
FPR[FPR_index-179]
TPR[FPR_index-179]
cutoff[FPR_index-179]

#################################TRAINING PERFORMANCE#################################
# Predicting dependent variable
preds_bst <- predict(xgbst, train_boost_x)
# Making a prediction object to caluclate TPR, and FPR
pred_full <- prediction(preds_bst, train_boost_y)
# Making a ROC object
roc_full <- performance(pred_full, "tpr", "fpr")
# Storing TPR, FPR, and cut-off values
TPR<-roc_full@y.values
FPR<-roc_full@x.values
cutoff<-roc_full@alpha.values
summary(FPR)
#revise the list to vector
FPR<-unlist(FPR)
TPR<-unlist(TPR)
cutoff<-unlist(cutoff)
# Finding the FPR index for FPR nearest to 0.1
FPR_index<-which.min(abs(FPR-0.1))
FPR_index
FPR[FPR_index]
TPR[FPR_index]
cutoff[FPR_index]

###########################################################################################
###########################STEP 7 (Final STEP): Making final file##########################
###########################################################################################

# Predicting perfect_booking_score for the test data
preds_bst <- predict(xgbst, airbnb_test)
# Classifying the predictions based on the results above
classifications_perfect <- ifelse(preds_bst > cutoff[FPR_index-179], "YES", "NO")
summary(as.factor(classifications_perfect))
# Writing the perfect_booking_score to a file
write.table(classifications_perfect, "perfect_rating_score_group23_9.csv", row.names = FALSE)


###########################################################################################
######################################LEARNING CURVE#######################################
###########################################################################################
# Setting up the number of training samples
training_cutoffs <- c(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                      20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000)
# Make a vector to store all the TPR values
TPR_list = rep(0,length(training_cutoffs))


for(index in c(1:length(training_cutoffs))){
  # TO know the status of the loop
  print(index)
  # sample our dataset to have limited data for training
  sampled_nrows <- sample(nrow(airbnb_train), 
                          training_cutoffs[index],
                          replace = FALSE)
  # Sampled dependent and independent variables
  sampled_airbnb_train <- airbnb_train[sampled_nrows, ]
  sampled_train_y <- train_y[sampled_nrows, ] 
  

  # Sampling the sampled_airbnb_train to create train and valid data
  valid_instn = sample(nrow(sampled_airbnb_train), 0.30*nrow(sampled_airbnb_train))
  # train_boost_x <- training x data.......valid_boost_x <- validation x data
  # train_boost_y <- training dependent data.......valid_boost_y <- validation dependent variables
  train_boost_x <- sampled_airbnb_train[-valid_instn, ]
  train_boost_y <- sampled_train_y$perfect_rating_score[-valid_instn]
  valid_boost_x <- sampled_airbnb_train[valid_instn, ]
  valid_boost_y <- sampled_train_y$perfect_rating_score[valid_instn]
  
  # Running the model with tuned hyperparameters
  xgbst <- xgboost(data = train_boost_x, label = train_boost_y,
                   max.depth = 15, eta = 0.05, nrounds = 2000,  
                   objective = "binary:logistic", verbosity = 0, verbose = FALSE)
  # Predicting dependent variable
  preds_bst <- predict(xgbst, valid_boost_x)
  # Making a prediction object to caluclate TPR, and FPR
  pred_full <- prediction(preds_bst, valid_boost_y)
  # Making a ROC object
  roc_full <- performance(pred_full, "tpr", "fpr")
  # Storing TPR, FPR, and cut-off values
  TPR<-roc_full@y.values
  FPR<-roc_full@x.values
  cutoff<-roc_full@alpha.values
  #revise the list to vector
  FPR<-unlist(FPR)
  TPR<-unlist(TPR)
  FPR_index<-which.min(abs(FPR-0.1))
  # Storing the required TPR in the vector
  TPR_list[index] <- TPR[FPR_index]
}

# plotting the learning curve
plot(training_cutoffs, TPR_list, type="o", col="blue", ylim=c(min(TPR_list),max(TPR_list)))

###########################################################################################
######################################FITTING CURVE########################################
###########################################################################################
# Setting up the number of training samples
nrounds_list <- c(20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 2500)
# Make a vector to store all the TPR values
TPR_list = rep(0,length(nrounds_list))

for(index in c(1:length(TPR_list))){
  # TO know the status of the loop
  print(index)
  # Sampling the sampled_airbnb_train to create train and valid data
  valid_instn = sample(nrow(airbnb_train), 0.30*nrow(airbnb_train))
  # train_boost_x <- training x data.......valid_boost_x <- validation x data
  # train_boost_y <- training dependent data.......valid_boost_y <- validation dependent variables
  train_boost_x <- airbnb_train[-valid_instn, ]
  train_boost_y <- train_y$perfect_rating_score[-valid_instn]
  valid_boost_x <- airbnb_train[valid_instn, ]
  valid_boost_y <- train_y$perfect_rating_score[valid_instn]
  
  # Running the model with tuned hyperparameters
  xgbst <- xgboost(data = train_boost_x, label = train_boost_y,
                   max.depth = 15, eta = 0.05, nrounds = 2000,  
                   objective = "binary:logistic", verbosity = 0, verbose = FALSE)
  # Predicting dependent variable
  preds_bst <- predict(xgbst, valid_boost_x)
  # Making a prediction object to caluclate TPR, and FPR
  pred_full <- prediction(preds_bst, valid_boost_y)
  # Making a ROC object
  roc_full <- performance(pred_full, "tpr", "fpr")
  # Storing TPR, FPR, and cut-off values
  TPR<-roc_full@y.values
  FPR<-roc_full@x.values
  cutoff<-roc_full@alpha.values
  #revise the list to vector
  FPR<-unlist(FPR)
  TPR<-unlist(TPR)
  FPR_index<-which.min(abs(FPR-0.1))
  # Storing the required TPR in the vector
  TPR_list[index] <- TPR[FPR_index]
}
# plotting the learning curve
plot(nrounds_list, TPR_list, type="o", col="blue", ylim=c(min(TPR_list),max(TPR_list)))

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
