#load libraries
library(tidyverse)
library(caret)
library(text2vec)
library(tm)
library(SnowballC)
library(vip)
library(textdata)
library(tidytext)
library(quanteda)
library(xgboost)
library(ROCR)
#load data files
train_x <- read_csv("airbnb_train_x_2023.csv")
train_y <- read_csv("airbnb_train_y_2023.csv")
test_x <- read_csv("airbnb_test_x_2023.csv")

#add a column for marking training and test data (train/test)

train_x<-train_x%>%
  mutate(mark = "trian")

test_x<-test_x%>%
  mutate(mark = "test")

airbnb<-rbind(train_x, test_x)

#feature engneering

airbnb$price <- as.numeric(gsub("\\$", "", airbnb$price))
airbnb$cleaning_fee <- as.numeric(gsub("\\$", "", airbnb$cleaning_fee))

# Replace NAs in cleaning_fee and price with 0
airbnb <- airbnb %>% 
  mutate(cleaning_fee = ifelse(is.na(cleaning_fee),0,cleaning_fee))

# Replace NAs in other numerical variables with their mean

airbnb <- airbnb %>% 
  mutate(price = ifelse(is.na(price),mean(price, na.rm=TRUE),price))

# Asides from price,bedrooms, beds, host_total_listings_count are also numerical variables

airbnb <- airbnb %>% 
  mutate(bedrooms =ifelse(is.na(bedrooms),mean(bedrooms,na.rm=TRUE),bedrooms ))

airbnb <- airbnb %>% 
  mutate(beds =ifelse(is.na(beds),mean(beds,na.rm=TRUE),beds))

airbnb <- airbnb %>% 
  mutate(host_total_listings_count=ifelse(is.na(host_total_listings_count),mean(host_total_listings_count,na.rm=TRUE),host_total_listings_count))

# price_per_person is the nightly price per accommodates 
airbnb <- airbnb %>% 
  mutate(price_per_person = price/accommodates)

#  has_cleaning_fee is YES if there is a cleaning fee, and NO otherwise
airbnb <- airbnb %>% 
  mutate(has_cleaning_fee= ifelse(cleaning_fee==0, 'No','Yes'))

airbnb$has_cleaning_fee<-as.factor(airbnb$has_cleaning_fee)
#  bed_category is "bed" if the bed_type is Real Bed and "other" otherwise
airbnb <- airbnb %>% 
  mutate(bed_category = ifelse(bed_type=='Real Bed','Bed','Others'))

airbnb$bed_category<-as.factor(airbnb$bed_category)

# property_category has the following values:
# apartment if property_type is Apartment, Serviced apartment, Loft.
# hotel if property_type is Bed & Breakfast, Boutique hotel, Hostel.
# condo if property_type is Townhouse, Condominium.
# house if property_type is Bungalow, House.
# other, otherwise

#creat vetors to store different variables

apartment <-c('Apartment','Serviced apartment','Loft')
hotel <-c('Bed & Breakfast','Boutique hotel',' Hostel')
condo <-c('Townhouse','Condominium')
house <-c('Bungalow','House')

airbnb <- airbnb %>% 
  mutate(property_category = ifelse(property_type %in% apartment, "apartment",
                                    ifelse(property_type %in% hotel, "hotel",
                                           ifelse(property_type %in% condo, "condo",
                                                  ifelse(property_type %in% house, "house", "other")))))


airbnb$property_category = as.factor(airbnb$property_category)
# ppp_ind is 1 if the price_per_person is greater than the median for the 
# property_category, and 0 otherwise

airbnb <- airbnb %>%
  group_by(property_category)%>%
  mutate(ppp_ind = ifelse(price_per_person>median(price_per_person),1,0))

# convert different variables to factors
airbnb$ppp_ind<-as.factor(airbnb$ppp_ind)

airbnb$cancellation_policy<-as.factor(airbnb$cancellation_policy)

airbnb$room_type<-as.factor(airbnb$room_type)


# Replace NAs in bathrooms with the median value
airbnb <- airbnb %>% 
  mutate(bathrooms =ifelse(is.na(bathrooms),mean(bathrooms,na.rm=TRUE),bathrooms))

# Replace NAs in host_is_superhost with FALSE
airbnb <- airbnb %>% 
  mutate(host_is_superhost =ifelse(is.na(host_is_superhost),FALSE,host_is_superhost))

# Create a new (factor) variable called "charges_for_extra" which has the value "YES" if extra_people > 0 and "NO" if extra_people is 0 or NA

airbnb$extra_people = as.numeric(gsub("\\$", "", airbnb$extra_people))

airbnb<-airbnb%>%
  mutate(charges_for_extra = ifelse(extra_people>0,'YES','NO'))

airbnb$charges_for_extra <- as.factor(airbnb$charges_for_extra)

#Create a new (factor) variable called "host_acceptance" from host_acceptance_rate with the values "ALL" if host_acceptance_rate = 100%, "SOME" if host_acceptance_rate < 100%, and "MISSING" if it's NA.

airbnb$host_acceptance <- ifelse(is.na(airbnb$host_acceptance_rate), "MISSING",
                                 ifelse(airbnb$host_acceptance_rate == "100%", "ALL", "SOME"))

airbnb$host_acceptance <- as.factor(airbnb$host_acceptance)

# Similarly, create a new (factor) variable called "host_response" with the values "ALL" if host_response_rate = 100%, "SOME" if host_response_rate < 100%, and "MISSING" if it's NA.

airbnb$host_response <- ifelse(is.na(airbnb$host_response_rate), "MISSING",
                               ifelse(airbnb$host_response_rate == "100%", "ALL", "SOME"))

airbnb$host_response <- as.factor(airbnb$host_response)

# Create a new factor variable called "has_min_nights" which is "YES" if minimum_nights > 1, and "NO" otherwise

airbnb<-airbnb%>%
  mutate(has_min_nights = ifelse(minimum_nights>1,'YES','NO'))

airbnb$has_min_nights<-as.factor(airbnb$has_min_nights)

#Replace market with "OTHER" if there are under 300 instances in that market. Convert market to a factor.

airbnb<-airbnb%>%
  group_by(market)%>%
  mutate(market = ifelse(n()<300,'OTHER',market))

airbnb$market<-as.factor(airbnb$market)


# added two more colums of list_till_today last_review_till_today 
airbnb<-airbnb%>%
  mutate(list_till_today=Sys.Date()-host_since,
         last_review_till_today=Sys.Date()-first_review,
         list_till_today = as.numeric(list_till_today),
         last_review_till_today = as.numeric(last_review_till_today))

# By executing following code we can get top 15 amendities 
# df_am<-data.frame(airbnb$amenities)
# 
# sum(is.na(df_am))
# 
# df_am$airbnb.amenities<-ifelse(df_am$airbnb.amenities=="","No Data",df_am$airbnb.amenities)
# 
# install.packages("fastDummies")
# library(fastDummies)
# 
# valid_instn = sample(nrow(df_am), 0.001*nrow(df_am))
# df_am_test<-df_am[valid_instn,]
# 
# dummy_df <- dummy_cols(
#   df_am_test,
#   remove_first_dummy = FALSE,
#   remove_most_frequent_dummy = FALSE,
#   ignore_na = TRUE,
#   split = ",",
#   remove_selected_columns = FALSE
# )
# 
# dummy_df<-dummy_df[,2:ncol(dummy_df)]
# amendity_sum<-colSums(dummy_df)
# 
# amendity_summary<-data.frame(amendity_sum)
# amendity_summary<-cbind(amendity = row.names(amendity_summary),amendity_summary)
# amendity_summary <- amendity_summary[order(amendity_summary$amendity_sum,decreasing = TRUE),]
# row.names(amendity_summary) <- NULL
# amendity_top15 <- head(amendity_summary, n = 15)
# 
# amendity_top15$amendity <- gsub(".data_", "",amendity_top15$amendity)
# amendity_top15$amendity

#Execute above codes to get Top 15 Amendity:
#"essentials"                "kitchen"                   "tv"                       
#"shampoo"                   "hangers"                   "hair dryer"               
#"washer"                    "dryer"                     "wireless internet"        
#"laptop friendly workspace" "iron"                      "internet"                 
#"family kid friendly"       "free parking on premises"  "fire extinguisher"


# creat dummy variables for top 15 amendities
airbnb$amenities <- gsub("\\{|\\}", "", airbnb$amenities)
airbnb$amenities <- gsub("\"", "", airbnb$amenities)

airbnb<-airbnb%>%
  mutate(has_detector = as.factor(grepl("detector", amenities)),
         has_Internet = as.factor(grepl("Internet", amenities)),
         has_TV = as.factor(grepl("TV", amenities)),
         has_friendly = as.factor(grepl("friendly", amenities)),
         has_Heating = as.factor(grepl("Heating", amenities)),
         has_Kitchen = as.factor(grepl("Kitchen", amenities)),
         has_Essentials= as.factor(grepl("Essentials", amenities)),
         has_Smoke = as.factor(grepl("Smoke", amenities)),
         has_Air = as.factor(grepl("Air", amenities)),
         has_Shampoo = as.factor(grepl("Shampoo", amenities)),
         has_conditioning = as.factor(grepl("conditioning", amenities)),
         has_Hangers = as.factor(grepl("Hangers", amenities)),
         has_Carbon = as.factor(grepl("Carbon", amenities)),
         has_Dryer = as.factor(grepl("Dryer", amenities)),
         has_Washer = as.factor(grepl("Washer", amenities)),
         has_Hair = as.factor(grepl("Hair", amenities)),
         has_monoxide = as.factor(grepl("monoxide", amenities)),
         has_Laptop = as.factor(grepl("Laptop", amenities)),
         has_Iron = as.factor(grepl("Iron", amenities)),
         has_dryer = as.factor(grepl("dryer", amenities)))

# sentiment analysis for "rules" column to know whether the rule is positive or negetive

cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    #removeWords(stopwords(kind="en")) %>% #remove stopwords
    #stemDocument %>%
    word_tokenizer 
}


# Iterate over the individual documents and convert them to tokens
# Uses the functions defined above.
it_train = itoken(airbnb$house_rules, 
                  preprocessor = tolower, 
                  tokenizer = cleaning_tokenizer, 
                  progressbar = FALSE)

# Step 3: Create the vocabulary from the tokenized itoken object
vocab = create_vocabulary(it_train)

vocab_small = prune_vocabulary(vocab, vocab_term_max = 500)

#Step 4: Vectorize (convert into DTM)
# Create a vectorizer object using the vocabulary we learned
vectorizer = vocab_vectorizer(vocab_small)

# Convert the training documents into a DTM and make it a binary BOW matrix
dtm_train = create_dtm(it_train, vectorizer)
dim(dtm_train)
dtm_train_bin <- dtm_train>0+0

bing_negative <- get_sentiments("bing") %>%
  filter(sentiment == 'negative')

bing_positive <- get_sentiments("bing") %>%
  filter(sentiment == 'positive')

mydict <- dictionary(list(negative = bing_negative$word, positive = bing_positive$word))

# convert the dtm into a dfm (document feature matrix)
dfm_train <- as.dfm(dtm_train)

sentiments <- dfm_lookup(dfm_train, mydict, valuetype = 'fixed')

sentiments <- convert(sentiments, to = "data.frame") %>%
  mutate(sent_score = as.factor(ifelse(positive >= negative, 'P', 'N')))

#added sentimentate analysis back to airbnb df

airbnb <- cbind(airbnb, rule_sent_score = sentiments$sent_score)


# change the host_response_time to different factors
airbnb$host_response_time<-ifelse(is.na(airbnb$host_response_time),"No record",airbnb$host_response_time)
airbnb$host_response_time<-as.factor(airbnb$host_response_time)

# change the instant_bookable to factors
airbnb$instant_bookable<-as.factor(airbnb$instant_bookable)


# change zipcode to factor
airbnb$zipcode<-as.factor(airbnb$zipcode)

# # added external data by using zipcode with population (not helpful)
# population <- read_csv("uszips.csv")
# 
# names(population)[names(population) == "zip"] <- "zipcode"
# 
# summary(population$zipcode)
# 
# population$zipcode<-as.character(population$zipcode)
# 
# population<-population%>%
#   select(zipcode,population)
# 
# airbnb <- merge(airbnb,population, by = "zipcode", all.x = TRUE)
# 
# summary(airbnb$population)
# #deal with NA value
# median_pop<-median(airbnb$population,na.rm = TRUE)
# airbnb$population<-
#   ifelse(is.na(airbnb$population),median_pop,airbnb$population)




airbnb_select<-airbnb%>%
  select(accommodates,bedrooms,cancellation_policy, 
         has_cleaning_fee,host_total_listings_count,price,ppp_ind,property_category,
         bathrooms,charges_for_extra,host_acceptance,has_min_nights,market,host_is_superhost,room_type,list_till_today,last_review_till_today,rule_sent_score,host_response_time,instant_bookable,
         has_Internet,has_TV,has_friendly ,has_Heating ,has_Kitchen ,has_Essentials,has_Smoke ,has_Air ,has_Shampoo ,has_conditioning ,has_Hangers ,has_Carbon ,
         has_Dryer ,has_Washer ,has_Hair ,has_monoxide,has_Laptop ,has_Iron,has_dryer)

# create dummy variable for airbnb df
dummy <- dummyVars( ~ . , data=airbnb_select, fullRank = TRUE)
airbnb_select<- data.frame(predict(dummy, newdata = airbnb_select))

#add mark column back to spearate train and test data
airbnb_select<-airbnb_select%>%
  mutate(mark = airbnb$mark)

#sperate the data for training and testing
train_perfect <-filter(airbnb_select, mark == "trian")
test_x<-filter(airbnb_select, mark == "test")

#remove mark column
train_perfect <-train_perfect%>%
  select(-mark)
train_perfect<-cbind(train_perfect,train_y)
train_perfect<-train_perfect%>%
  select(-high_booking_rate)

test_x<-test_x%>%
  select(-mark)

#get target variable
#train_y<-train_y$perfect_rating_score
#sum(is.na(train_y))


#finished data preparation

#start to train model
set.seed(1)
## Partition 30% of the data as validation data
valid_instn = sample(nrow(train_perfect), 0.30*nrow(train_perfect))

# set training and validation data: x
train_log<-train_perfect[-valid_instn,]
valid_log<-train_perfect[valid_instn,]

#train a linear regression model

#formula of regression model
formula_log<-perfect_rating_score ~ .

# function of prediction
tr_pred <- function(train_data, valid_data, model_formula){
  trained_model <- lm(data = train_data, model_formula, family = "binomial") 
  predictions <- predict(trained_model, newdata = valid_data, type = "response") 
  return(predictions)
}

train_log$perfect_rating_score<-as.factor(train_log$perfect_rating_score)
valid_log$perfect_rating_score<-as.factor(valid_log$perfect_rating_score)
probs<-tr_pred(train_log, valid_log, formula_log)

probs<-ifelse(is.na(probs),0,probs)
pred_full <- prediction(probs,valid_log$perfect_rating_score)


roc_full <- performance(pred_full, "tpr", "fpr")
plot(roc_full, col = "red", lwd = 2)


TPR<-roc_full@y.values

FPR<-roc_full@x.values

cutoff<-roc_full@alpha.values
summary(FPR)
#revise the list to vector
FPR<-unlist(FPR)
TPR<-unlist(TPR)
cutoff<-unlist(cutoff)
#find FPR close to 0.1
FPR_index<-which.min(abs(FPR-0.1))
FPR_index
FPR[FPR_index]
TPR[FPR_index]


####### Training set Performance

train_log$perfect_rating_score<-as.factor(train_log$perfect_rating_score)
valid_log$perfect_rating_score<-as.factor(valid_log$perfect_rating_score)
probs<-tr_pred(train_log, train_log, formula_log)

probs<-ifelse(is.na(probs),0,probs)
pred_full <- prediction(probs,train_log$perfect_rating_score)


roc_full <- performance(pred_full, "tpr", "fpr")
plot(roc_full, col = "red", lwd = 2)


TPR<-roc_full@y.values

FPR<-roc_full@x.values

cutoff<-roc_full@alpha.values
summary(FPR)
#revise the list to vector
FPR<-unlist(FPR)
TPR<-unlist(TPR)
cutoff<-unlist(cutoff)
#find FPR close to 0.1
FPR_index<-which.min(abs(FPR-0.1))
FPR_index
FPR[FPR_index]
TPR[FPR_index]
