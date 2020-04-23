#!/usr/local/bin/python3.7

import urwid
import mido 
import sys
import atexit
import traceback
import bisect
from time import time
from collections import deque
import math

# you might want to change this:
INPUTS = ['monologue #2 KBD/KNOB']
INPUTS = mido.get_input_names()

OUTPUTS = mido.get_output_names()
OUTPUTS = sorted(OUTPUTS, key=lambda s: s.lower())

# specific to INPUT
TICK_SIZE = 6


class Step: 
  def __init__(self, seq, step):
    self.seq = seq
    self.msg = None
    self.len = TICK_SIZE
    self.div = 1
    self.skip = False

  def on(self):
    if self.msg:
      if self.skip:
        self.skip = False
        return False
      else: 
        self.seq.send(self.msg)
        return True

  def off_promise(self):
    if self.msg and self.msg.type == 'note_on':
      note = self.msg.note
      return lambda: self.seq.send(mido.Message('note_off', note=note))
    else:
      return lambda: None

  def set(self, msg, len, div, mummify=True, skip=False):
    if mummify:
      self.mummify()

    if msg is None:
      msg = self.msg

    if len is None:
      len = self.len

    if div is None: 
      div = self.div 

    self._set(msg, len, div)

    self.skip = skip

  def _set(self, msg, len, div):
    self.msg = msg
    self.len = len
    self.div = div

  def copy(self, other):
    self.mummify()
    self._set(other.msg, other.len, other.div)

  def clear(self): 
    self.mummify()
    self._set(None, TICK_SIZE, 1)

  def mummify(self): 
    msg = self.msg
    len = self.len
    div = self.div
    self.seq.undos.append(lambda: self._set(msg, len, div))

class Seq:
  def __init__(self, mae, port_name):
    self.steps = [Step(self, s) for s in range(mae.steps)]
    self.port_name = port_name
    print(f"open({port_name})")
    self.port_out = mido.open_output(port_name)
    self.undos = deque()
    self.mode = 'r'
    self.cc = None

  def clear(self):
    for note in self.steps:
      note.clear()

    self.port_out.reset()

    if not mae.playing:
      mae.reset_step()
      mae.inc_pos(-max(0, mae.selection[0]))

    self.cc = None

  def send(self, msg):
    self.port_out.send(msg)

    while len(mae.midi_msgs) > 12:
      mae.midi_msgs.popleft()

    mae.midi_msgs.append("+%s %s << %s" % (mae.tick, self.port_name, msg))

  def schedule(self, ticks, callable):
    if ticks > 0:
      bisect.insort(mae.event_q, Event(mae.tick + ticks, callable))
    else:
      callable()

  def play(self):
    if not mae.playing:
      return 

    if self.mode == 'm' and mae.solo is not self:
      return 

    if mae.solo and mae.solo is not self:
      return 

    note = self.steps[mae.step]

    if note.msg and note.on():
      self.schedule(note.len, note.off_promise())

      if note.div > 1: 
        gap = note.len / note.div

        r = [(n+1) * gap for n in range(note.div-1)]
        info(f"gap={gap} range={r}")
       
        for t in r:
          self.schedule(t, note.off_promise())
          self.schedule(t, note.on)

  def toggle_mute(self):
    if self.mode == 'm':
      self.mode = '-'
    else:
      self.mode = 'm' 
      self.port_out.reset()

  def toggle_rec(self):
    self.mode = 'r' if self.mode != 'r' else '-'

  def extend(self, steps):
    curr = len(self.steps)
    for s in range(curr, curr + steps):
      self.steps.append(Step(self, s))
    info(f"extend({steps})")

class Event:
  def __init__(self, tick, fun):
    self.tick = tick
    self.fun = fun

  def __lt__(self, other):
    return self.tick < other.tick

  def __gt__(self, other):
    return self.tick > other.tick

  def __str__(self):
    return f"@{self.tick}"

class CBuf:
  def __init__(self, len):
    self.buf = [ None for x in range(len) ]
    self.len = len
    self.pos = 0

  def add(self, value):
    self.buf[ self.pos % self.len] = value
    self.pos += 1
  
  def __len__(self):
    return self.len if self.pos >= self.len else self.pos

  def __iter__(self):
    return (v for v in self.buf if v is not None)

tempos = CBuf(8)

def on_tick(msg): 
  mae.tick += 1

  while mae.event_q and mae.event_q[0].tick <= mae.tick: 
    e = mae.event_q.popleft()
    e.fun()
    if e.tick - mae.tick >= 1:
      info(f"late: {e}")

  if not mae.playing: 
    return

  if mae.tick % TICK_SIZE == 0:
    for seq in mae.seqs:
      seq.play()          

    if mae.tick % 24 == 0:
      now = time() 
      mae.tempo = 1 / (now - mae.last_tick_time) * 60
      tempos.add(mae.tempo)
      mae.last_tick_time = now

    mae.inc_step()
    refresh()

def lpd8_hook(msg):
  lpd8 = None
  valu = None

  if msg.type == "program_change":
    if msg.channel == 0:
      lpd8 = lpd8cmds.get(f"pc{msg.program}")

  elif msg.type == "control_change":
    if msg.channel == 0:
      lpd8 = lpd8cmds.get(f"cc{msg.control}")
      valu = msg.value

  if lpd8:
    lpd8(valu)

  else:
    info(f"?{msg}")

note_on_ticks = {}

def on_msg(msg, name):
  if name == "LPD8":
    lpd8_hook(msg)
    refresh()
    return 

  try:
    if msg.type == "note_on":
      mae.seq_sel().send(msg)

      if mae.seq_sel().mode == 'r':
        # freeze?
        if mae.seq_sel().cc is not True:
          note = mae.note(mae._seq_sel, mae.pos)
          note.set(msg=msg, len=None, div=None, mummify=True, skip=mae.playing and mae.follow)
          note_on_ticks[ msg.note ] = (mae.tick, note)
          mae.seq_sel().cc = False

        else:
          info("cowardly refusing to record cc into a note track")

        if not mae.playing and mae.follow: 
          mae.inc_step()

      refresh()

    elif msg.type == "clock":
      on_tick(msg)

    elif msg.type == "note_off":
      mae.seq_sel().send(msg)

      start, note = note_on_ticks.pop(msg.note, (None, None))

      if start:
        len = mae.tick - start
        note.set(msg=None, len=len, div=None, mummify=False)

    elif msg.type == "control_change":
      mae.seq_sel().send(msg)

      if mae.seq_sel().mode == 'r':
        if mae.seq_sel().cc is not False:
          mae.note(mae._seq_sel, mae.pos).set(msg=msg, len=None, div=None, skip=mae.playing and mae.follow)
          mae.seq_sel().cc = True
          refresh()

        else:
          info("cowardly refusing to record notes into a cc track")

    elif msg.type == "start":
      mae.reset_step()
      mae.playing = True
      refresh()

    elif msg.type == "stop":
      mae.stop()
      refresh()

    else:
      info(f"?:{msg}")
  
  except Exception as e:
    info(f"{time()} error in midi loop: {traceback.format_exc()}")

def fmt_txt():
  buf = list()

  def quadrate(step_range, dot_fun, eol="\n", gap_fun=lambda step:"  "):
    for step in range(step_range[0], step_range[1]): 
      if step % 8 == 0: 
        buf.append(gap_fun(step))

      note = mae.note(q, step) # q? 
      buf.append(str(dot_fun(note, step)))

    buf.append(gap_fun(step_range[1]))
    buf.append(eol)

  def note_chr(note, step):
    if note.msg: 
      msg_sel = mae.note(q, mae.step).msg
      if msg_sel and note.msg.type == "note_on" and msg_sel.type == "note_on" and note.msg.note == msg_sel.note: 
        return u"\u25a3" 

      else:
        return u"\u25a0" 

    else:
      return u"\u25a1"

  def range_marks(seqNo):
    def xx(step): 
      if seqNo != mae._seq_sel:
        return "  "
      if step == mae.selection[0]:
        return "[ " 
      if step == mae.selection[1]:
        return "] " 
      return "  "
    return xx

  def stride(npl):
    s = 0 
    while s < mae.steps:
      yield [s, min(s+npl, mae.steps)]
      s += npl

  npl = 96 # notes, per, line

  for q in range(len(mae.seqs)):
    for step_range in stride(npl):

      quadrate(step_range, lambda _, step: "_ " if step == mae.step else "  ")

      quadrate(step_range, note_chr, eol="", gap_fun=range_marks(q))

      if not step_range[0]:
        if mae.solo == mae.seqs[q]: 
          buf.append("S")

        else:
          buf.append(mae.seqs[q].mode)

        buf.append(" ")
        buf.append(mae.seqs[q].port_name)

      buf.append("\n")

      quadrate(step_range, lambda _, step: "^ " if q == mae._seq_sel and step == mae.pos else "  ")

  note = mae.note(mae._seq_sel, mae.pos) 

  buf.append("play=%d " % (1 if mae.playing else 0))
  buf.append("step=%-5s " % f"{mae.step+1}:{mae.steps}")
  buf.append("pos=%-2s " % mae.pos)
  buf.append("follow=%d " % (1 if mae.follow else 0))
  buf.append("q=%-2s " % len(mae.event_q))
  buf.append("tempo=%s " % (("%-5.2f" % mae.tempo) if mae.tempo else '-'))

  lenx = len(tempos)
  if lenx > 1:
    mean = sum(tempos) / lenx
    sd = math.sqrt(sum((x - mean) * (x - mean) for x in tempos)/(lenx-1))
    buf.append("%1.2f=" % sd)
    buf.append(u"\u03c3")

    ms = int((sd / mae.tempo) * 60 * 1000)
    buf.append("=%sms" % ms)

  buf.append("\n")

  if note.msg and note.msg.type == "note_on":
    buf.append("%-2s " % NOTES[ (note.msg.note +3) %12 ])

  else:
    buf.append("-  ")

  buf.append("%s " % note.msg if note.msg else "-")

  if note.msg and note.msg.type == "note_on":
    buf.append("xlen=%s/%d " % (note.len, note.div))

  buf.append("\n\n")

  for i in range(len(mae.midi_msgs)-1, 0, -1):
    buf.append(mae.midi_msgs[i])
    buf.append("\n")

  buf.append("\n")
  buf.append(mae.info_txt)
  buf.append("\n")
  buf.append("\n")

  cols = 4
  cc = [ (k,v) for k, v in commands.items() ]

  def idx_order():
    rows = int(len(cc)/cols) + 1
    for i in range(cols):
      for j in range(rows):
        idx = i + j * cols
        if idx < len(cc):
          yield idx

  x=0
  for i in idx_order():
    x += 1
    k, v = cc[i]
    buf.append("%10s: %-10s " % (k, v[0]))
    if x and (x % cols) == 0:
      buf.append("\n")

  buf.append("\n")
  buf.append(str(list(idx_order())))

  return ''.join(buf)

def info(txt):
  mae.info_txt = str(txt)
  refresh()

def refresh():
  main.set_text(fmt_txt())
  loop.draw_screen()

commands = {}
lpd8cmds = {}

def register(key, desc, fun, lpd8=None):
  if key in commands:
    raise Exception(key)
  commands[key] = [desc, fun]
  if lpd8 is not None:
    if lpd8 in lpd8cmds:
      raise Exception(ldp8)
    lpd8cmds[lpd8] = fun

def register_knob(lpd8, fun):
  if lpd8 in lpd8cmds:
    raise Exception(ldp8)
  lpd8cmds[lpd8] = fun

def input(key):
  try:
    commands[key][1](None) 
    refresh()
  except KeyError: 
    info(f"?:{key}")
  except urwid.ExitMainLoop as e:
    raise e
  except Exception as e:
    info(f"error: {traceback.format_exc()}")

def enter(_):
  if mae.playing:
    mae.stop()
  else:
    mae.reset_step()
    mae.playing = True

register("enter", "play/stop", enter, "pc0")

register("r", "rec", lambda _: mae.seq_sel().toggle_rec(), "pc1")

def space(_):
  mae.note(mae._seq_sel, mae.pos).clear()
  if not mae.playing and mae.follow:
    mae.inc_step()

def xpose(low, hi, fun): 
  return lambda value: fun(int(value / 128.0 * (hi - low + 1) + low))

register(" ", "rest", space)

register("up", "seq-1", lambda _: mae.seq_sel(-1))

register("down", "seq+1", lambda _: mae.seq_sel(+1)) 

register("left", "step-1", lambda _: mae.inc_pos(-1))

register("right", "step+1", lambda _: mae.inc_pos(+1))

register("tab", "step>>", lambda _: mae.inc_pos(8 - (mae.pos % 8)), "pc3")

register("shift tab", "step>>", lambda _: mae.inc_pos(-1 - ((mae.pos - 1) % 8)), "pc2")

register("home", "step=0", lambda _: mae.reset_step()) 

register("[", "set-start", lambda _: mae.select(start=(mae.pos - (mae.pos % 8))))

register("]", "set-end", lambda _: mae.select(end=(mae.pos + (8 - mae.pos % 8))))

register("f", "follow", lambda _: mae.toggle_follow())

def backspace(_):
  if mae.step: 
    mae.step -= 1
  mae.curr().clear()

register("backspace", "step--", backspace)

def undo(_):
  try:
    mae.seq_sel().undos.pop()()
    info(f"undo({len(mae.seq_sel().undos)})")
  except IndexError:
    pass

register("u", "seq-undo", undo); 

register("m", "seq-mute", lambda _: mae.seq_sel().toggle_mute())

register("c", "seq-clear", lambda _: mae.seq_sel().clear()) 

register("s", "seq-solo", lambda _: mae.toggle_solo()) 

def throw_notes(_): 
  steps = mae.selection[1] - mae.selection[0]
  for i in range(steps):
    try:
      mae.copy_note(mae.selection[0] + i, mae.pos + i)
    except IndexError:
      return
  mae.inc_pos(steps)

register("/", "note-throw", throw_notes)

def note_len(inc):
  len = max(0, mae.selected().len + inc)
  mae.selected().set(msg=None, len=len, div=None)

register("l", "note.len+1", lambda _: note_len(+1));
register("L", "note.len-1", lambda _: note_len(-1));
register_knob("cc1", lambda val: mae.selected().set(msg=None, len=val, div=None))

def note_div(inc):
  note = mae.selected()
  if note.div + inc > 0:
    note.set(msg=None, len=None, div=note.div + inc)

register("d", "note.div", lambda _: note_div(+1))
register("D", "note.div", lambda _: note_div(-1))
register_knob("cc5", xpose(1, 128, lambda val: mae.selected().set(msg=None, len=None, div=val)))

def note_swipe():
  for pos in range(mae.pos-1, -1, -1):
    note = mae.note(mae._seq_sel, pos)
    if note.msg:
      mae.copy_note(pos, mae.pos)
      return

register("x", "->note", note_swipe)

register("p", "panic", lambda _: (seq.port_out.reset() for seq in mae.seqs))

def nudge(dir):
  if dir is 1:
    for s in range(mae.steps-2, mae.pos-1, -1):
      mae.copy_note(s, s+1)
    mae.note(mae._seq_sel, mae.pos).clear()
    mae.inc_pos(+1)

  elif dir is -1:
    if mae.step is not 0:
      for s in range(mae.pos, mae.steps):
        mae.copy_note(s, s-1)
      mae.note(mae._seq_sel, mae.steps-1).clear()
      mae.inc_pos(-1)

  else:
    raise Exception(dir)

register("meta b", "<nudge", lambda _: nudge(-1))

register("meta f", ">nudge", lambda _: nudge(+1))

def inc_pitch(inc, note): 
  if not note.msg:
    return

  if note.msg.type == "note_on":
    note.set(msg=note.msg.copy(note=(note.msg.note + inc) % 128), len=None, div=None)

  elif note.msg.type == "control_change":
    note.set(note.msg.copy(value=(note.msg.value + inc) % 128), len=None, div=None)
 
register("n",   "note+1", lambda _: inc_pitch(+1, mae.note(mae._seq_sel, mae.pos)))

register("N", "note-1", lambda _: inc_pitch(-1, mae.note(mae._seq_sel, mae.pos)))

def inc_pitch_range(inc): 
  for n in range(mae.selection[0], mae.selection[1]): 
    inc_pitch(inc, mae.note(mae._seq_sel, n))

register("t", "sel-tpose", lambda _: inc_pitch_range(+1))

register("T", "sel-tpose", lambda _: inc_pitch_range(-1))

def inc_velo(inc): 
  note = mae.note(mae._seq_sel, mae.pos) 
  if not note.msg:
    return

  if note.msg.type == "note_on":
    note.set(note.msg.copy(velocity=(note.msg.velocity + inc) % 128), len=None, div=None)

register("v", "velo+1", lambda _: inc_velo(+1))

register("V", "velo-1", lambda _: inc_velo(-1))

register("i", "+seq", lambda _: mae.insert_seq())

register("I", "-seq", lambda _: mae.del_seq())

register("e", "seq+8", lambda _: mae.extend(8))

def exit(_):
  raise urwid.ExitMainLoop()

register("q", "quit", exit)

class Maestro:
  def __init__(self, steps):
    self.steps = steps
    self.step = 0
    self.selection = [0, steps]
    self.playing = False
    self.tick = 0 
    self.last_tick_time = time()
    self._seq_sel = 0
    self.pos = 0
    self.info_txt = "-"
    self.midi_msgs = deque()
    self.tempo = None
    self.seqs = [Seq(self, port) for port in OUTPUTS]
    self.event_q = deque()
    self.solo = None
    self.follow = True

  def seq_sel(self, inc=0):
    self._seq_sel = (self._seq_sel + inc) % len(self.seqs)
    return self.seqs[self._seq_sel]

  def curr(self):
    return self.note(self._seq_sel, self.step)

  def selected(self):
    return self.note(self._seq_sel, self.pos)

  def inc_step(self):
    self.step = (self.step + 1) % min(self.steps, self.selection[1])
    if self.step == 0:
      self.reset_step()
    if self.follow:
      self.pos = self.step

  def inc_pos(self, inc):
    self.pos = (self.pos + inc) % self.steps
    if mae.playing:
      mae.follow = False
    if mae.follow:
      mae.step = mae.pos

  def note(self, seq, step):
    return self.seqs[seq].steps[step]

  def stop(self):
    self.playing = False
    for e in self.event_q:
      e.fun()
    self.event_q.clear()

  def reset_step(self):
    self.step = max(0, self.selection[0])
    if self.follow or not self.playing:
      self.pos = self.step

  def select(self, start=None, end=None):
    if start is not None:
      if start == self.selection[0]:
        self.selection[0] = 0

      elif start >= self.selection[1]: 
        width = self.selection[1] - self.selection[0]
        self.selection[1] = min(self.steps, start + width)
        self.selection[0] = start 

      else:
        self.selection[0] = start 

    if end is not None:
      if end == self.selection[1]:
        self.selection[1] = self.steps 

      elif end <= self.selection[0]:
        width = self.selection[1] - self.selection[0]
        self.selection[0] = max(0, end - width)
        self.selection[1] = end

      else:
        self.selection[1] = end 

    info(f"select({self.selection[0]},{self.selection[1]})")

  def toggle_follow(self):
    self.follow = not self.follow

  def toggle_solo(self):
    if self.solo == self.seq_sel():
      self.solo = None

    else:
      self.solo = self.seq_sel()

      for seq in self.seqs:
        if seq != self.solo:
          seq.port_out.reset()
  
  def copy_note(self, step0, stepN):
    src = mae.seq_sel().steps[step0]
    dst = mae.seq_sel().steps[stepN]
    dst.copy(src)

  def insert_seq(self):
    seq = Seq(self, self.seq_sel().port_name)
    self.seqs.insert(self._seq_sel+1, seq)
    self.seq_sel(+1)

  def del_seq(self):
    if len(self.seqs) == 1:
      raise urwid.ExitMainLoop()
    del self.seqs[self._seq_sel]
    self.seq_sel(-1)

  def extend(self, steps):
    for seq in self.seqs:
      seq.extend(steps)
    self.steps += steps

mae = Maestro(64) 

NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

main = urwid.Text(fmt_txt())
fill = urwid.Filler(main, 'top')
loop = urwid.MainLoop(fill, unhandled_input=input)
urwid.set_encoding("UTF-8")

port_ins = {}
for input in INPUTS: 
  print(f"open({input})")
  def wrap(name__): 
    return lambda msg: on_msg(msg, name__)    
  port_ins[ input ] = mido.open_input(input, callback=wrap(input))

def shutdown():
  print("bye")

  for port, input in port_ins.items(): 
    print(f"close({port})")
    input.close()

  for seq in mae.seqs:
    print(f"close({seq.port_name})")
    seq.port_out.panic()
    seq.port_out.close()

  print("bye")

#atexit.register(shutdown)

try:
  loop.run()
except Exception as e:
  info(f"error in main loop: {traceback.format_exc()}")

shutdown()
