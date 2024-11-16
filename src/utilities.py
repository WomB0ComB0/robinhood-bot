#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os

from dotenv import load_dotenv

load_dotenv()


class TwitterCredentials:
    def __init__(self):
        self.consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
        self.consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")

    @property
    def empty_credentials(self):
        """Returns True is any credential is empty; False otherwise"""

        return not (bool(self.consumer_key) and bool(self.consumer_secret))


class RobinhoodCredentials:
    def __init__(self):
        self.user = os.getenv("ROBINHOOD_USER")
        self.password = os.getenv("ROBINHOOD_PASS")
        self.mfa_code = os.getenv("ROBINHOOD_MFA_CODE")
        print(f"Loaded credentials: User={self.user}, MFA Code={self.mfa_code}")

    @property
    def empty_credentials(self):
        """Returns True is any credential is empty; False otherwise"""

        return not (bool(self.user) and bool(self.password) and bool(self.mfa_code))
