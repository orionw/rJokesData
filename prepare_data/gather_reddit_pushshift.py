"""
Code framework taken from:
    https://www.osrsbox.com/blog/2019/03/18/watercooler-scraping-an-entire-subreddit-2007scape/
"""

import requests
import json
import re
import time
import datetime
import os
from dateutil.relativedelta import relativedelta

PUSHSHIFT_REDDIT_URL = "http://api.pushshift.io/reddit"

def fetchObjects(**kwargs):
    # Default paramaters for API query
    params = {
        "sort_type":"created_utc",
        "sort":"asc",
        "size":1000,
        "filter": ("id", "created_utc")
        }

    # Add additional paramters based on function arguments
    for key,value in kwargs.items():
        params[key] = value

    # Print API query paramaters
    print(params)

    # Set the type variable based on function input
    # The type can be "comment" or "submission", default is "comment"
    type_post = "comment"
    if 'type' in kwargs and kwargs['type'].lower() == "submission":
        type_post = "submission"
    
    # Perform an API request
    r = requests.get(PUSHSHIFT_REDDIT_URL + "/" + type_post + "/search/", params=params, timeout=30)

    # Check the status code, if successful, process the data
    if r.status_code == 200:
        response = json.loads(r.text)
        data = response['data']
        sorted_data_by_id = sorted(data, key=lambda x: int(x['id'],36))
        return sorted_data_by_id
    
    else:
        return []

def extract_reddit_data(**kwargs):
    # Speficify the start timestamp, make sure we get everything
    max_created_utc = int((datetime.datetime.now() - relativedelta(years=100)).timestamp())
    max_id = 0

    # Open a file for JSON output
    file = open(os.path.join(os.path.realpath('..'), "data", "submissions.json"), "w")

    # While loop for recursive function
    while 1:
        nothing_processed = True
        # Call the recursive function
        print("After {}".format(max_created_utc))
        objects = fetchObjects(**kwargs,after=max_created_utc)
        
        # Loop the returned data, ordered by date
        for object in objects:
            id = int(object['id'], 36)
            if id > max_id:
                nothing_processed = False
                created_utc = object['created_utc']
                max_id = id
                if created_utc > max_created_utc: max_created_utc = created_utc
                # Output JSON data to the opened file
                print(json.dumps(object, sort_keys=True, ensure_ascii=True), file=file)
        
        # Exit if nothing happened
        if nothing_processed: 
            return
        max_created_utc -= 1

        # Sleep a little before the next recursive function call
        time.sleep(.5)

# Start program by calling function with:
# 1) Subreddit specified
# 2) The type of data required (comment or submission)
extract_reddit_data(subreddit="jokes",type="submission")