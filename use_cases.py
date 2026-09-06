USE_CASES = {
    1: {
        "title": "Customer Churn Prediction",
        "icon": "bi-person-dash",
        "context": (
            "A telecom company wants to know, in advance, which customers are "
            "likely to cancel their subscription (churn) so it can act before "
            "they leave."
        ),
        "data": [
            "Customer tenure (months as a subscriber)",
            "Monthly charges and contract type",
            "Number of support tickets opened",
            "Service usage patterns (calls, data, streaming)",
        ],
        "target": (
            "Classify each customer as \"will churn\" or \"will not churn\" "
            "in the near future."
        ),
        "ml_type": "Supervised Learning &mdash; Classification",
        "benefit": (
            "The company can proactively offer retention incentives to "
            "at-risk customers, reducing revenue loss and improving customer "
            "lifetime value."
        ),
    },
    2: {
        "title": "House Price Estimation",
        "icon": "bi-house-door",
        "context": (
            "A real-estate platform wants to estimate a fair market price "
            "for a house before it is listed, based on its physical "
            "characteristics and location."
        ),
        "data": [
            "Property size in square meters",
            "Number of rooms and bathrooms",
            "Neighborhood / location",
            "Age of the property",
        ],
        "target": (
            "Predict a continuous value: the estimated sale price of the house."
        ),
        "ml_type": "Supervised Learning &mdash; Regression",
        "benefit": (
            "Buyers, sellers and agents get a fast, data-driven price "
            "reference instead of relying only on manual appraisals."
        ),
    },
    3: {
        "title": "Customer Segmentation for Marketing",
        "icon": "bi-people",
        "context": (
            "A retail company wants to group its customers by purchasing "
            "behavior so it can design targeted marketing campaigns instead "
            "of a single generic campaign for everyone."
        ),
        "data": [
            "Purchase frequency",
            "Average ticket size per purchase",
            "Product categories usually bought",
            "Recency of the last purchase",
        ],
        "target": (
            "Discover hidden groups (clusters) of customers with similar "
            "behavior; there is no predefined label to predict."
        ),
        "ml_type": "Unsupervised Learning &mdash; Clustering",
        "benefit": (
            "Marketing campaigns become more personalized and relevant, "
            "improving conversion rates and return on marketing spend."
        ),
    },
    4: {
        "title": "Warehouse Robot Navigation",
        "icon": "bi-robot",
        "context": (
            "A warehouse operator wants an autonomous robot to learn the "
            "best path to pick up items and deliver them while avoiding "
            "obstacles and minimizing travel time."
        ),
        "data": [
            "Sensor readings (distance to obstacles, walls, shelves)",
            "Current position and orientation",
            "Rewards or penalties received after each action",
            "Layout of the warehouse (states the robot can be in)",
        ],
        "target": (
            "Learn a sequence of actions (a policy) that maximizes the "
            "cumulative reward, rather than a single prediction."
        ),
        "ml_type": "Reinforcement Learning",
        "benefit": (
            "The robot adapts to changes in the warehouse layout and "
            "improves its efficiency over time, reducing operating costs."
        ),
    },
}
