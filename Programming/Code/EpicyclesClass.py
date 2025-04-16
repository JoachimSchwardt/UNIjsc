"""
EpicyclesClass WIP, missing documentation on the following parts:
    self.run_track()
    file.__doc__
"""
import cv2
import numpy as np
import pygame
import time
import sys
# from numba import njit


    
# # higher performance using numba? not working as of now

# @njit(cache = True)
# def _coef_njit(track, n):
#     track_complex = np.empty(track.shape[0])
#     tl = np.shape(track)[0]
#     track_complex = track[:, 0] + 1j * track[:, 1]
#     k_array = np.arange(n, -1-n, -1, dtype=np.int64)
#     kt_matrix = np.outer(k_array, np.arange(0, tl, 1, dtype=np.int64))
#     exp_array = np.exp(2*np.pi*1j * kt_matrix / tl) * track_complex
#     return np.sum(exp_array, axis=1) / tl

class Epicycles(object):
    def __init__(self, track=[], ftrack=[], colors=[], coef=[], 
                 res=(1920, 1080), n=10, decay=0.6, Auto_process_track=True, 
                 Auto_run_track=True):
        """
        Initialize the class. All parameters are optional.
            'colors': Array like object of length 5:
                0: Handdrawn lines (default to white)
                1: Replicated Fourier-drawing (default to yellow)
                2, 3, 4: Background lines (default to gray-shades)
            'res': Tuple with two entries, 'width' and 'heigth' in pixels.
            'n': Approximation is done with '2*n + 1' circles / frequencies.
            'decay': Float between 0 and 1, describes how quickly the Fourier-
                     track fades to black .
            'track': Handdrawn track, can instead be given as an Array like
                     object with arbitrary length. Each element must be a 
                     tuple of the x and y-coordinates of a point in the track.
            'ftrack': Replica of 'track' by Fourier Epicycles, is created
                      while running the track. 
            'coef': Coefficients of the needed Fourier Series. Can be given as
                    an Array like object of length '2*n + 1'.
        """
        if len(colors) != 5:      # use default colors if none are given
            white = pygame.Color(220, 220, 220)
            yellow = pygame.Color(255, 255, 0)
            gray = pygame.Color(120, 120, 120)
            dark_gray = pygame.Color(90, 90, 90)
            black = pygame.Color(30, 30, 30)
            self.colors = [white, yellow, gray, dark_gray, black]
        else:
            self.c = colors         # set of five colors, see description 
        [self.w, self.h] = res      # screen resolution (width by heigth)
        self.n = n                  # order of approximation
        self.decay = max(0, min(1, decay))
        self.track = track
        self.ftrack = ftrack
        self.coef = coef
        
        # Whether to auto-process or auto-run a drawn or loaded track
        self.Auto_process_track = Auto_process_track
        self.Auto_run_track = Auto_run_track
        
        # Initialize the window with the 'pygame' library
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        
        # Initialize the use of pressed keys while the program is running
        self.keys = pygame.key.get_pressed() # Currently uses key.events()!
        
    def start_text(self, start_text_str):
        """
        Displays the given text on screen. Parameter must be a string.
        """
        text = self.font.render(start_text_str, True, self.c[0])
        self.screen.blit(text, ((self.w - text.get_width()) // 2, 
                                (self.h - text.get_height()) // 2))
        pygame.display.update()
        pass
    
    def draw_grid(self):
        """
        Draws grid lines on the background. Color names in comments are 'c':
            Colors taken from the 'self.c' Array, used indices are 2, 3, 4.
        """
        self.screen.fill(self.c[4])        # color the background with c4
        
        # draw vertical and horizontal line with c2 at the screen center
        pygame.draw.line(self.screen, self.c[2], (self.w//2, 0), 
                         (self.w//2, self.h))
        pygame.draw.line(self.screen, self.c[2], (0, self.h//2), 
                         (self.w, self.h//2))
        
        # draw the grid lines with c3
        for k in range(50, self.w//2, 50):  # vertical lines
            pygame.draw.line(self.screen, self.c[3], (self.w//2 + k, 0), 
                             (self.w//2 + k, self.h))
            pygame.draw.line(self.screen, self.c[3], (self.w//2 - k, 0), 
                             (self.w//2 - k, self.h))
        for k in range(50, self.h//2, 50):  # horizontal lines
            pygame.draw.line(self.screen, self.c[3], (0, self.h//2 + k), 
                             (self.w, self.h//2 + k))
            pygame.draw.line(self.screen, self.c[3], (0, self.h//2 - k), 
                             (self.w, self.h//2 - k))
            
    def decision(self):
        """
        Allows for switching between different modes via button pressing:
            'Q': Quit.
            'Space': Start recording a handdrawn track.
            'P': Process the 'self.track'.
            'R': Run the Animation generated by 'self.coef'.            
        """
        while True:
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    self.close()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    self.record_track()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_p:
                    self.process_track()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    self.run_track()
                    
    def close(self):
        """
        Closes the display and terminates the script with 'sys.exit'.
        """
        pygame.display.quit()
        pygame.quit()
        sys.exit()      
        
        
    def record_track(self):
        """
        Initiated by pressing the Spacebar. First point of the handdrawn track
            is the mouse-position at that instance. Freely move around the
            mouse to draw the track. 
            Recording is finished by pressing 'Space' again.
        """
        print("Recording Track ...")
        
        self.draw_grid()            # reset screen to default grid
        self.track = [pygame.mouse.get_pos()]   # starting point
        
        # record the track until 'Space' is preessed again
        wait = True    
        while wait:
            time.sleep(0.01)    # Time-resolution of the recording in seconds
            for e in pygame.event.get():        # check for Spacebar
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    wait = False                # finish drawing track
            
            x0, y0 = self.track[-1]             # previous track point
            x1, y1 = pygame.mouse.get_pos()     # get mouse position
            d = ((x1 - x0)**2 + (y1 - y0)**2)**0.5   # distance 'new' to 'old'
            
            # append the pixels in a straigth line between 'new' and 'old'
            for i in range(2, int(d), 2):
                new_vals = [x0 + (x1-x0) * i/d, y0 + (y1-y0) * i/d]
                self.track.append(new_vals)
            
            # display the drawn track with the color c0
            for [x, y] in self.track:
                self.screen.set_at((int(x), int(y)), self.c[0])
            pygame.display.update()
        
        # move the drawn track to match with what is displayed on screen
        self.track = np.array(self.track) - [self.w//2, self.h//2]
        if self.Auto_process_track == True:
            self.process_track()    # automatically process the track
        else:
            self.decision()         # back to idle mode (decision tree)
    
    def process_track(self):
        """
        Compute the first '2*n + 1' coefficients 'c_k' in the Fourier series.
            1. Convert 2D-track to a 1D-Array of complex numbers 
                ('x'-values correspond to real-part)
            2. Create a 1D-Array of integers corresponding to the frequencies
            3. Construct the tensor product with the 1D-Array of indices of
                the track-Array
            4. Do the complex exponential of this 2D-Array and multiply with
                the complex track
            5. Sum along the axis of the track, approximating the integral. 
                This contracts the Matrix to a 1D-Array of '2*n + 1' coef.
        """
        print("Processing Track ...")
        t_start = time.time()
        tl = len(self.track)
        track_complex = [complex(x, y) for [x, y] in self.track]
        k_array = np.arange(self.n, -1-self.n, -1)
        kt_matrix = np.outer(k_array, np.arange(tl))
        exp_array = np.exp(2*np.pi*1j * kt_matrix / tl) * track_complex
                     
        self.coef = np.sum(exp_array, axis=1) / tl
        t_end = time.time()
        print("Processing Done in {} s!".format(t_end - t_start))
        if self.Auto_run_track == True:
            self.run_track()        # automatically run the track
        else:
            self.decision()         # back to idle mode (decision tree)
        
    def _alternating_integers(self):
        """
        Returns a 1D-Array of alternating integers, meaning the output is
            [0, 1, -1, 2, -2, 3, ..., self.n, -self.n]
        """
        ans = np.zeros(2*self.n + 1, dtype=int)
        x = np.arange(1, self.n + 1, 1, dtype=int)
        ans[1::2] = x
        ans[2::2] = -x
        return ans
        
    def run_track(self):
        tl = len(self.track)
        if tl == 0:
            print("No track given, drawing from coefficients only...")
            tl = 1000             # default track length, precision vs. speed
        if len(self.coef) != 2*self.n + 1:
            self.process_track()  # avoid IndexError in self.coef
        print("Running Track!")
        
        # implementation of the actual loop creating the visuals
        alternating_integers = self._alternating_integers()
        for t in range(tl * 100):
            self.screen.fill(self.c[4])
            z = complex(self.w//2, self.h//2)
            # for k in sum(zip(range(1, self.n + 1, 1), 
            #                  range(-1, -1-self.n, -1)), (0,)):
            for k in alternating_integers:
                old_z = z
                z += self.coef[k + self.n] * np.exp(2*np.pi*1j * k * t / tl)
                
                # connect the 'old_z' to 'z' with line
                old_coords = (int(old_z.real), int(old_z.imag))
                coords = (int(z.real), int(z.imag))
                pygame.draw.line(self.screen, self.c[2], old_coords, coords)
                
                # draw circle at 'z_old' if it is larger than 2 pixels
                r = abs(z - old_z)
                if r > 1:
                    pygame.draw.circle(self.screen, self.c[3],
                                       old_coords, int(r), 1)
            
            # recreate 'track' with 'ftrack' by adding the calculated 'z'-vals
            if len(self.ftrack) < tl:
                self.ftrack.append(z)
            
            # light path with decay modifier -> line fades over time
            for k, p in enumerate(self.ftrack):
                color_mult = (1 - self.decay * ((t - k) % tl) / tl)
                color = [self.c[1][j] * color_mult for j in range(3)]
                
                self.screen.set_at((int(p.real), int(p.imag)), color)
            
            
            # apply changes 
            pygame.display.update()
            
            # slightly convoluted implementation of a menu during the run
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    self.close()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_e:
                    print("Back to Menu")
                    self.ftrack = []
                    self.decision()
                # stop the run on button press
                if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                    wait = True     
                    while wait:
                        time.sleep(0.1)
                        for se in pygame.event.get():
                            if (se.type == pygame.KEYDOWN 
                                and se.key == pygame.K_s):
                                wait = False        # continue the run
                            if (se.type == pygame.KEYDOWN 
                                and se.key == pygame.K_q):
                                self.close()


def contours(image, Sort_By_Area=False, Sort_By_Length=False):  
    """
    Image recognition with the 'cv2' package. Attempts to find the outer most
        closed contour in a 2D-Image for replicating it using Fourier series.
    """
    img = cv2.imread(image, 0)      # load image
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    if Sort_By_Area == True and Sort_By_Length == False:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    elif Sort_By_Length == True and Sort_By_Area == False:
        cnts = sorted(cnts, key=len, reverse=True)[0]
    return cnts


def main():
    print(__doc__)
    white = pygame.Color(220, 220, 220)     # handdrawn line color
    yellow = pygame.Color(255, 255, 0)      # fourier line color
    gray = pygame.Color(120, 120, 120)      # mostly rotating circle color
    dark_gray = pygame.Color(90, 90, 90)    # mostly gridline color
    black = pygame.Color(30, 30, 30)        # background color
    colors = [white, yellow, gray, dark_gray, black]
    
    n = 100               # order of the approximation, '2*n + 1' circles used
    res = (1920, 1080)    # resolution of the created window in pixels
    decay = 0.6           # decay parameter for the replicated line (0 to 1)
    start_text_str = ("Press SPACE to Start/Stop drawing, P to Process, "
                      +"R to Run, E to Exit from Run, Q to Quit")
    track = []
    # xvals = np.linspace(-300, 300, 1000)
    # yvals = np.zeros(len(xvals))
    # track = list(zip(xvals, yvals))
    ftrack = []
    coef = []
    
    # image = "ClosedCurveExample.png"
    # image = "DragonExample.png"
    image = "BlackWhiteDragon.jpg"
    track = contours(image, False, True)    # load image and find contour
    
    # resize the track to fit the screen
    track = np.array([[int(x / 3), int(y / 3)] for [[x, y]] in track])
    track = track - np.sum(track, axis=0) / len(track)   # center track
    track = track[np.arange(0, len(track), 10)]          # thin out the track
    # track = []
    # coef = [100, 0, 100]
    
    # Initiate the screen as an object with inbuilt function calls
    EPI = Epicycles(track=track, ftrack=ftrack, colors=colors, coef=coef,
                    res=res, n=n, decay=decay, Auto_process_track=True)
    EPI.draw_grid()
    EPI.start_text(start_text_str)
    EPI.decision()
    
    # numba test, insignificant speed up
    # t1 = time.perf_counter()
    # _coef_njit(EPI.track, EPI.n)
    # t2 = time.perf_counter()
    # print(t2 - t1)
    
    
if __name__ == "__main__":
    main()