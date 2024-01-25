# %%
import pandas as pd

# create a dummy table of users
users = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 5],
        "user_name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        "title": ["Dr.", "Mr.", "Ms.", "Mr.", "Mrs."],
    }
).set_index("user_id")

# create a dummy table of posts
posts = pd.DataFrame(
    {
        "post_id": [1, 2, 3, 4, 5],
        "post_title": [
            "My first post",
            "My second post",
            "My third post",
            "My fourth post",
            "My fifth post",
        ],
        "post_body": [
            "This is my first post. I hope you like it!",
            "This is my second post. I hope you like it!",
            "This is my third post. I hope you like it!",
            "This is my fourth post. I hope you like it!",
            "This is my fifth post. I hope you like it!",
        ],
        "user_id": [1, 1, 2, 3, 3],
    }
).set_index("post_id")

# %%
users

# %%
posts

# %%
# define a schema for linking users to posts
user_post_schema = {
    "validate": "one_to_many",
    "table1": "users",
    "table2": "posts",
    "index1": "user_id",
    "col1": None,
    "index2": None,
    "col2": "user_id",
}


class FrameLink:
    def __init__(self, table1, table2, schema):
        self.table1 = table1
        self.table2 = table2
        self.schema = schema

    def validate(self):
        how = self.schema["validate"]
        if how == "one_to_many":
            self._validate_one_to_many()
        else:
            raise ValueError(f"invalid validation method: {how}")
    
    # self._