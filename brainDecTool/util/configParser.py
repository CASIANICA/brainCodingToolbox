# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import ConfigParser

class Config:
    def __init__(self, path):
        """Config instance initialization."""
        self.path = path
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(self.path)

    def get(self, field, key):
        """Get config value."""
        try:
            result = self.cf.get(field, key)
        except:
            result = ''
        return result

    def set(self, field, key, value):
        """Set config value."""
        try:
            self.cf.set(field, key, value)
            cf.write(open(self.path, 'w'))
        except:
            return False
        return True

