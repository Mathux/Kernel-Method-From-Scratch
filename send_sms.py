from json import dumps
import sys
from urllib3.request import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError

def send_sms(text="Empty!", secured=True):
    """ Sens a free SMS to the user identified by [user], with [password].
    :user: Free Mobile id (of the form [0-9]{8}),
    :password: Service password (of the form [a-zA-Z0-9]{14}),
    :text: The content of the message (a warning is displayed if the message is bigger than 480 caracters)
    :secured: True to use HTTPS, False to use HTTP.
    Returns a boolean and a status string.
    """

    #: Identification Number free mobile
    user = '17575827'

    #: Password
    password = 'STtYs1YlsuL9Gw'

    print("Your message is: ", text)
    dictQuery = {"user": user, "pass": password, "msg": text}
    url = "https" if secured else "http"
    string_query = dumps(dictQuery, sort_keys=True, indent=4)
    string_query = string_query.replace(password, '*' * len(password))
    print("\nThe web-based query to the Free Mobile API (<u>{}://smsapi.free-mobile.fr/sendmsg?query<U>) will be based on:\n{}.".format(url, string_query))

    query = urlencode(dictQuery)
    url += "://smsapi.free-mobile.fr/sendmsg?{}".format(query)

    try:
        urlopen(url)
        return 0
    except HTTPError as e:
        if hasattr(e, "code"):
            return e.code
        else:
            print("Unknown error...")
            return 2, "Unknown error..."