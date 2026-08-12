"""Training loop.

Written for short, interruptible free-tier sessions: checkpoint every epoch, resume from the
last checkpoint, never assume the session survives to the end of the schedule.
"""
