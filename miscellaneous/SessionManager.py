import os
import datetime
import glob


class SessionManager:
    def __init__(self, parent_directory):
        if not os.path.isdir(parent_directory):
            os.mkdir(parent_directory)
        self.parent_dir = parent_directory
        self.session_dir = None

    def get_subdirectories(self):
        subdirectories = glob.glob(os.path.join(self.parent_dir, '*/'))
        return subdirectories

    def create_dir(self, name=None):
        """Create session folder with given name or current date/time"""
        if name is None:
            now = datetime.datetime.now()
            name = now.strftime("%Y-%m-%d-%H-%M-%S")
        # Set to current session folder and create folder
        self.session_dir = os.path.join(self.parent_dir, name)
        os.mkdir(self.session_dir)
        return

    def get_most_recent(self):
        """Returns most recently created session folder"""
        most_recent_dir = max(glob.glob(os.path.join(self.parent_dir, '*/')), key=os.path.getmtime)
        return most_recent_dir

    def set_session_most_recent(self):
        self.session_dir = self.get_most_recent()

    def get_fpath(self, fname):
        return os.path.join(self.session_dir, fname)

    def files_in_session_dir(self):
        files_in_session = [os.path.join(self.session_dir, f)
                            for f in os.listdir(self.session_dir)
                            if os.path.isfile(os.path.join(self.session_dir, f))
                            ]
        return files_in_session
