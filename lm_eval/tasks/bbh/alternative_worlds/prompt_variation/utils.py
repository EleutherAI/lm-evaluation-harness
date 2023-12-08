# For fixing line 163 in `movie_recommendation`

def fix_movie_recommendation(data):

    def _fix(doc):
        if doc["target"] == "Monsters, Inc":
            doc["input"] = "Find a movie similar to Minority Report, Shrek, Catch Me If You Can, Aladdin:\nOptions:\n(A) Monsters, Inc\n(B) Children of the Night\n(C) The Incredible Shrinking Man\n(D) Town & Country"
            doc["target"] = "(A)"
        return doc

    return data.map(_fix)

def fix_ruin_names(data):

    def _fix(doc):
        if doc["target"] == "dearth, wind, & fire":
            doc["input"] = "Which of the following is a humorous edit of this artist or movie name: 'earth, wind, & fire'?\nOptions:\n(A) eareth, wind, & fire\n(B) earth, bind, & fire\n(C) earthm wind, & fire\n(D) dearth, wind, & fire"
            doc["target"] = "(D)"

        elif doc["target"] == "rita, sue and bob poo":
            doc["input"] = "Which of the following is a humorous edit of this artist or movie name: 'rita, sue and bob too'?\nOptions:\n(A) rita, sue and bob too\n(B) rita, sue and bob poo\n(C) rita, sue and box too\n(D) rita,y sue and bob too"
            doc["target"] = "(B)"
        return doc

    return data.map(_fix)