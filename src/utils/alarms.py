import pygame
import os


class AlarmPlayer:
    def __init__(self, sound_file=None):
        pygame.mixer.init()
        self.sound = None
        self.playing = False

        if sound_file and os.path.exists(sound_file):
            self.sound = pygame.mixer.Sound(sound_file)
        else:
            print(f"⚠️ Fichier audio introuvable: {sound_file}")
            # Créer un son de bip simple si pas de fichier
            self._create_beep()

    def _create_beep(self):
        """Crée un bip simple si pas de fichier audio"""
        # Son de 1 seconde à 440 Hz (note La)
        import numpy as np
        sample_rate = 22050
        duration = 0.5
        frequency = 440

        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)
        wave = (wave * 32767).astype(np.int16)

        # Stéréo
        stereo_wave = np.column_stack((wave, wave))
        self.sound = pygame.sndarray.make_sound(stereo_wave)

    def start(self):
        """Démarre l'alarme en boucle"""
        if self.sound and not self.playing:
            self.sound.play(-1)  # -1 = boucle infinie
            self.playing = True

    def stop(self):
        """Arrête l'alarme"""
        if self.sound and self.playing:
            self.sound.stop()
            self.playing = False

    def __del__(self):
        """Nettoie les ressources"""
        self.stop()