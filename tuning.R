#Author: Gyan Shashwat
#Tuning Code with model

# model instantiation -----------------------------------------------
# set defaul flags
FLAGS <- flags(
  flag_numeric("dropout", 0.4), 
  flag_integer("unit1", 100),
  flag_integer("unit2", 100),
  flag_integer("unit3", 100),
  flag_numeric("lambda", 0.01)
  
)


# model configuration
model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$unit1, input_shape = V, activation = "relu", name = "layer_1",
              kernel_regularizer = regularizer_l2(FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = FLAGS$unit2, activation = "relu", name = "layer_2",
              kernel_regularizer = regularizer_l2(FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = FLAGS$unit3, activation = "relu", name = "layer_3",
              kernel_regularizer = regularizer_l2(FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = ncol(y_train), activation = "softmax", name = "layer_out") %>%
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_adam(),
  )

fit <- model %>% fit(
  x = xz_train, y = y_train,
  validation_data = list(x_val, y_val),
  epochs = 100,
  verbose = 1,
  batch_size = N *0.02, #fixed batch size for the operation
  callbacks = callback_early_stopping(monitor = "val_accuracy", patience = 20)
)

# store accuracy on test set for each run
score <- model %>% evaluate(
  xtun_test, ytun_test,
  verbose = 0
)