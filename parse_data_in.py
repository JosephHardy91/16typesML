__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import datetime
from tqdm import tqdm

print "Getting data"
typeLines = [x.split(",") for x in
             open("D:/Documents/PythonProjects/sixteentypesML/data/usernames.csv", "r").readlines()]
type_dictionary = dict([(x[1].lower().strip(), x[2].strip()) for x in typeLines[1:] if x[2].strip() != "???"])

post_dictionary = defaultdict(lambda: defaultdict(dict))
with open("D:/Documents/PythonProjects/sixteentypesML/data/scrape10152016.txt", "r") as scrapedData:
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

            if len(content.split(" ")) > 1:
                v = {'content': content}
                if not 'username' in post_dictionary[userID]:
                    post_dictionary[userID]['username'] = username
                if not 'type' in post_dictionary[userID] and post_dictionary[userID]['username'].lower() in type_dictionary:
                    post_dictionary[userID]['type'] = type_dictionary[
                        post_dictionary[userID]['username'].lower()]
                post_dictionary[userID]['posts'][dateTime] = v
        except:
            pass  # print post_lines
to_remove = []
for userID in post_dictionary:
    if 'type' not in post_dictionary[userID]:
        to_remove.append(userID)

for userID in to_remove:
    del post_dictionary[userID]
# for userID in post_dictionary:
#     most_common_name = Counter(
#         [post_dictionary[userID][post]['username'] for post in post_dictionary[userID]]).most_common()
#     post_dictionary[userID].update({'username': most_common_name})


# print "Getting types"
type_list = [post_dictionary[userID]['type'] for userID in post_dictionary if
             post_dictionary[userID]['username'].lower() in type_dictionary]
# for userID in post_dictionary.keys():

# for post_date in post_dictionary[userID]:
#     if post_dictionary[userID]['posts'][post_date]['username'].lower() in type_dictionary:
#         type_ = type_dictionary[post_dictionary[userID][post_date]['username'].lower()]
#         post_dictionary[userID].update({'username': post_dictionary[userID][post_date]['username'].lower(),
#                                         'type': type_})
#         type_list.append(type_)
#         break

unique_type_list = list(set(type_list))

# print post_dictionary[post_dictionary.keys()[0]]
