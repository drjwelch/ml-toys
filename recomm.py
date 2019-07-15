import numpy as np
import tensorrec

# Build the model with default parameters
model = tensorrec.TensorRec()

# Generate some dummy data
interactions, user_features, item_features = \
              tensorrec.util.generate_dummy_data( \
                  num_users=15,
                  num_items=6,
                  interaction_density=.4,
                  pos_int_ratio=1)
print("user")
print(user_features)
print()
print("item")
print(item_features)
print()
print("ranks")
print(interactions.todense())
print()

# Fit the model for 5 epochs
model.fit(interactions, user_features, item_features, epochs=50, verbose=True)

# Predict scores for all users and all items
print("ranks")
print(interactions.todense())
print()
predictions = model.predict(user_features=user_features,
                            item_features=item_features)
print("predictions")
print(predictions)

print(type(user_features))
