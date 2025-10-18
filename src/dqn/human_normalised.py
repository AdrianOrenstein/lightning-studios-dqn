import math

# from https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/atari_data.py


# Game: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: score human (float).
ATARI_DATA = {
    "ALE/Alien-v5": (227.8, 7127.7),
    "ALE/Amidar-v5": (5.8, 1719.5),
    "ALE/Assault-v5": (222.4, 742.0),
    "ALE/Asterix-v5": (210.0, 8503.3),
    "ALE/Asteroids-v5": (719.1, 47388.7),
    "ALE/Atlantis-v5": (12850.0, 29028.1),
    "ALE/BankHeist-v5": (14.2, 753.1),
    "ALE/BattleZone-v5": (2360.0, 37187.5),
    "ALE/BeamRider-v5": (363.9, 16926.5),
    "ALE/Berzerk-v5": (123.7, 2630.4),
    "ALE/Bowling-v5": (23.1, 160.7),
    "ALE/Boxing-v5": (0.1, 12.1),
    "ALE/Breakout-v5": (1.7, 30.5),
    "ALE/Centipede-v5": (2090.9, 12017.0),
    "ALE/ChopperCommand-v5": (811.0, 7387.8),
    "ALE/CrazyClimber-v5": (10780.5, 35829.4),
    "ALE/Defender-v5": (2874.5, 18688.9),
    "ALE/DemonAttack-v5": (152.1, 1971.0),
    "ALE/DoubleDunk-v5": (-18.6, -16.4),
    "ALE/Enduro-v5": (0.0, 860.5),
    "ALE/FishingDerby-v5": (-91.7, -38.7),
    "ALE/Freeway-v5": (0.0, 29.6),
    "ALE/Frostbite-v5": (65.2, 4334.7),
    "ALE/Gopher-v5": (257.6, 2412.5),
    "ALE/Gravitar-v5": (173.0, 3351.4),
    "ALE/Hero-v5": (1027.0, 30826.4),
    "ALE/IceHockey-v5": (-11.2, 0.9),
    "ALE/Jamesbond-v5": (29.0, 302.8),
    "ALE/Kangaroo-v5": (52.0, 3035.0),
    "ALE/Krull-v5": (1598.0, 2665.5),
    "ALE/KungFuMaster-v5": (258.5, 22736.3),
    "ALE/MontezumaRevenge-v5": (0.0, 4753.3),
    "ALE/MsPacman-v5": (307.3, 6951.6),
    "ALE/NameThisGame-v5": (2292.3, 8049.0),
    "ALE/Phoenix-v5": (761.4, 7242.6),
    "ALE/Pitfall-v5": (-229.4, 6463.7),
    "ALE/Pong-v5": (-20.7, 14.6),
    "ALE/PrivateEye-v5": (24.9, 69571.3),
    "ALE/Qbert-v5": (163.9, 13455.0),
    "ALE/Riverraid-v5": (1338.5, 17118.0),
    "ALE/RoadRunner-v5": (11.5, 7845.0),
    "ALE/Robotank-v5": (2.2, 11.9),
    "ALE/Seaquest-v5": (68.4, 42054.7),
    "ALE/Skiing-v5": (-17098.1, -4336.9),
    "ALE/Solaris-v5": (1236.3, 12326.7),
    "ALE/SpaceInvaders-v5": (148.0, 1668.7),
    "ALE/StarGunner-v5": (664.0, 10250.0),
    "ALE/Surround-v5": (-10.0, 6.5),
    "ALE/Tennis-v5": (-23.8, -8.3),
    "ALE/TimePilot-v5": (3568.0, 5229.2),
    "ALE/Tutankham-v5": (11.4, 167.6),
    "ALE/UpNDown-v5": (533.4, 11693.2),
    "ALE/Venture-v5": (0.0, 1187.5),
    # Note the random agent score on Video Pinball is sometimes greater than the
    # human score under other evaluation methods.
    "ALE/VideoPinball-v5": (16256.9, 17667.9),
    "ALE/WizardOfWor-v5": (563.5, 4756.5),
    "ALE/YarsRevenge-v5": (3092.9, 54576.9),
    "ALE/Zaxxon-v5": (32.5, 9173.3),
}

_RANDOM_COL = 0
_HUMAN_COL = 1

ATARI_GAMES = tuple(sorted(ATARI_DATA.keys()))


def get_human_normalized_score(game: str, raw_score: float = 0) -> float:
    if raw_score is None:
        return math.nan
    """Converts game score to human-normalized score."""
    game_scores = ATARI_DATA.get(game, (math.nan, math.nan))
    random, human = game_scores[_RANDOM_COL], game_scores[_HUMAN_COL]
    return (raw_score - random) / (human - random)
