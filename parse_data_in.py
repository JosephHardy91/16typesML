__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import datetime
from tqdm import tqdm

print "Getting data"
post_dictionary = defaultdict(lambda: defaultdict(dict))
with open("../data/scrape10152016.txt", "r") as scrapedData:
    textContent = scrapedData.read()
    posts = textContent.split("===============\n")[1:]
    for post in tqdm(posts):
        try:
            post_lines = post.split("\n")
            if len(post_lines[0].split()) > 1 or "=" in post_lines[0]:
                continue
            usernameline = 1
            creep = False
            if "starpoo" in post_lines[0]:
                usernameline = 8
            elif "Creepy" in post_lines[0]:
                creep = True
            if not creep:
                usernameUserId = post_lines[usernameline].strip()
            else:
                usernameUserId = post_lines[0].strip() + post_lines[1]
            userID, username = usernameUserId.split("-", 1)
            dateline = post_lines[usernameline + 1].strip()
            dateTime = datetime.datetime.strptime(dateline, "%m-%d-%Y,%H:%M %p")
            content = "\n".join(post_lines[usernameline + 2:])
            post_dictionary[userID][dateTime] = {'content': content, 'username': username}
        except:
            pass  # print post_lines

# for userID in post_dictionary:
#     most_common_name = Counter(
#         [post_dictionary[userID][post]['username'] for post in post_dictionary[userID]]).most_common()
#     post_dictionary[userID].update({'username': most_common_name})
typeLines = [x.split(",") for x in open("../data/usernames.csv", "r").readlines()]
type_dictionary = dict([(x[1].lower().strip(), x[2].strip()) for x in typeLines[1:] if x[2].strip() != "???"])

print "Getting types"
for userID in tqdm(post_dictionary.keys()):
    for post in post_dictionary[userID]:
        if post_dictionary[userID][post]['username'].lower() in type_dictionary:
            post_dictionary[userID].update({'username': post_dictionary[userID][post]['username'].lower(),
                                            'type': type_dictionary[post_dictionary[userID][post]['username'].lower()]})
            break
