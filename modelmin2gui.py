#!/usr/bin/env python3
"""
Attendance Optimizer - Fractional Simplex (single-file)

Requirements:
    - Python 3.8+
    - scipy
    - tkinter (bundled with Python on most systems)

Notes:
    - Uses tk.Label for colored result boxes so background/foreground work on macOS dark mode.
    - If there are no inequality/equality constraints, we pass None to linprog for those args
      to avoid dimension errors.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from scipy.optimize import linprog
import math

# -------------------------
# Data
# -------------------------
PROFESSORS = [
    {"id": 1, "name": "Prof. B. Biswal", "pv": 7.2, "le": 6.5, "se": 7.0, "ar": 5.5},
    {"id": 2, "name": "Prof. Shobha Bagai", "pv": 8.5, "le": 7.8, "se": 8.0, "ar": 9.2},
    {"id": 3, "name": "Prof. Pankaj Tyagi", "pv": 7.0, "le": 6.5, "se": 6.8, "ar": 8.5},
    {"id": 4, "name": "Prof. Swati Arora", "pv": 6.5, "le": 5.8, "se": 5.5, "ar": 7.0},
    {"id": 5, "name": "Prof. Mahima Kaushik", "pv": 7.8, "le": 7.2, "se": 7.5, "ar": 8.0},
    {"id": 6, "name": "Prof. Nirmal Yadav", "pv": 8.2, "le": 7.5, "se": 8.0, "ar": 8.8},
    {"id": 7, "name": "Prof. Sonam Tanwar", "pv": 7.5, "le": 6.8, "se": 7.2, "ar": 7.8},
    {"id": 8, "name": "Prof. Asani Bhaduri", "pv": 7.8, "le": 7.0, "se": 7.3, "ar": 8.2},
    {"id": 9, "name": "Prof. Harendra Pal Singh", "pv": 7.3, "le": 6.7, "se": 7.0, "ar": 7.5},
    {"id": 10, "name": "Prof. Sachin Kumar", "pv": 7.6, "le": 6.9, "se": 7.2, "ar": 7.9},
    {"id": 11, "name": "Prof. J.S. Purohit", "pv": 7.1, "le": 6.4, "se": 6.7, "ar": 7.3},
    {"id": 12, "name": "Prof. Dorje Dawa", "pv": 6.8, "le": 6.2, "se": 6.5, "ar": 7.0},
    {"id": 13, "name": "Prof. Shobha Rai", "pv": 7.4, "le": 6.8, "se": 7.1, "ar": 7.6},
    {"id": 14, "name": "Prof. Anjani Verma", "pv": 7.2, "le": 6.6, "se": 6.9, "ar": 7.4},
    {"id": 15, "name": "Prof. Manish Kumar", "pv": 7.7, "le": 7.1, "se": 7.4, "ar": 8.1},
    {"id": 16, "name": "Prof. Sanjeewani Sehgal", "pv": 7.5, "le": 6.9, "se": 7.2, "ar": 7.7},
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME_SLOTS = [
    {"id": 1, "time": "9:00-10:00", "block": "morning"},
    {"id": 2, "time": "10:00-11:00", "block": "morning"},
    {"id": 3, "time": "11:00-12:00", "block": "midday"},
    {"id": 4, "time": "12:00-13:00", "block": "midday"},
    {"id": 5, "time": "14:00-15:00", "block": "afternoon"},
    {"id": 6, "time": "15:00-16:00", "block": "afternoon"},
]

# -------------------------
# Helpers
# -------------------------
def find_professor(pid):
    return next((p for p in PROFESSORS if p["id"] == pid), None)

def calculate_aps(instructor_id, time_block, student_profile):
    """
    Returns APS score (a float) for a given instructor/timeblock/student.
    The weight coefficients are arbitrary but normalized.
    """
    prof = find_professor(instructor_id)
    if prof is None:
        return 0.0

    timeBlockRatings = {"morning": 7.5, "midday": 7.0, "afternoon": 6.5}
    # weights chosen to sum roughly to 1.0
    weights = {
        "w1": 0.1476, "w2": 0.1456, "w3": 0.1370, "w4": 0.1356,
        "w5": 0.1203, "w6": 0.1225, "w7": 0.0935, "w8": 0.0979
    }
    TB = timeBlockRatings.get(time_block, 7.0)
    HS = 5.0  # hypothetical holiday-skip baseline

    aps = (
        weights["w1"] * prof["pv"] +
        weights["w2"] * prof["le"] +
        weights["w3"] * prof["se"] +
        weights["w4"] * prof["ar"] +
        weights["w5"] * TB +
        weights["w6"] * HS -
        weights["w7"] * student_profile.get("travelTime", 0) -
        weights["w8"] * student_profile.get("timeCommitment", 0)
    )
    return round(aps, 4)

def optimize_attendance_fractional(timetable, student_profile, weeks=20, attendance_target=0.75):
    """
    Linear program to pick fractional attendance (0..1 per slot) to maximize APS sum
    while matching required fraction of weekly classes (attendance_target).
    Returns dict with results or None on failure.
    """
    total_weekly = len(timetable)
    if total_weekly == 0:
        return None

    # required per week (can be fractional)
    required_per_week = total_weekly * attendance_target
    # sanity clamp: cannot require more than available classes
    required_per_week = min(required_per_week, total_weekly)

    total_semester = total_weekly * weeks
    required_semester = int(math.ceil(required_per_week * weeks))

    # Build class_list preserving ordering for variable indexing
    class_list = []
    for key, cls in timetable.items():
        slot = next((s for s in TIME_SLOTS if s["id"] == cls["slotId"]), None)
        aps = calculate_aps(cls["instructorId"], slot["block"] if slot else "midday", student_profile)
        class_list.append({**cls, "aps": aps, "key": key})

    n = len(class_list)
    if n == 0:
        return None

    # Objective: maximize sum(aps * x) where 0 <= x <= 1.
    # linprog minimizes, so use -aps
    c = [-cls["aps"] for cls in class_list]

    # Equality constraint: sum(x_i) == required_per_week  (or we can allow small tolerance, but keep exact)
    A_eq = [[1.0] * n]
    b_eq = [required_per_week]

    # Instructor minimum constraint: for each instructor with >= 2 classes, ensure sum >= 2
    # Convert to A_ub * x <= b_ub form by multiplying by -1: -sum(indices) <= -2
    A_ub = []
    b_ub = []

    instructor_map = {}
    for idx, cls in enumerate(class_list):
        iid = cls["instructorId"]
        instructor_map.setdefault(iid, []).append(idx)

    for iid, indices in instructor_map.items():
        if len(indices) >= 2:
            row = [0.0] * n
            for i in indices:
                row[i] = -1.0
            A_ub.append(row)
            b_ub.append(-2.0)

    # If A_ub is empty, set to None to avoid shape issues
    A_ub_arg = A_ub if len(A_ub) > 0 else None
    b_ub_arg = b_ub if len(b_ub) > 0 else None
    A_eq_arg = A_eq if len(A_eq) > 0 else None
    b_eq_arg = b_eq if len(b_eq) > 0 else None

    # Bounds for each variable 0..1
    bounds = [(0.0, 1.0)] * n

    # Solve
    try:
        res = linprog(
            c=c,
            A_ub=A_ub_arg,
            b_ub=b_ub_arg,
            A_eq=A_eq_arg,
            b_eq=b_eq_arg,
            bounds=bounds,
            method='highs'
        )
    except Exception as e:
        # As fallback try without instructor constraints
        try:
            res = linprog(
                c=c,
                A_ub=None,
                b_ub=None,
                A_eq=A_eq_arg,
                b_eq=b_eq_arg,
                bounds=bounds,
                method='highs'
            )
        except Exception:
            return None

    if not res.success:
        # second fallback: allow inequality sum(x) >= required_per_week converted to -sum <= -required_per_week
        # but linprog doesn't support >=; to allow slack, we can try to relax equality to <= required_per_week
        # Try relaxing equality to inequality sum(x) >= required_per_week by using bounds to push?
        # For simplicity, return None if not successful
        return None

    # Build attendance fractions dict keyed by key
    attendance_fractions = {}
    for i, x in enumerate(res.x):
        cls = class_list[i]
        key = f"{cls['day']}-{cls['slotId']}"
        attendance_fractions[key] = {
            "fraction": float(round(x, 3)),
            "aps": float(cls["aps"]),
            "aps_weighted": float(round(x * cls["aps"], 3)),
            "day": cls["day"],
            "slotId": cls["slotId"],
            "instructorId": cls["instructorId"],
            "subject": cls.get("subject", "(no subject)")
        }

    total_frac_week = sum(v["fraction"] for v in attendance_fractions.values())
    total_selected_semester = total_frac_week * weeks
    attendance_percentage = 100.0 * (total_selected_semester / total_semester) if total_semester > 0 else 0.0
    total_value = sum(v["aps_weighted"] for v in attendance_fractions.values())
    avg_value = (total_value / total_frac_week) if total_frac_week > 0 else 0.0

    # Instructor stats
    instructor_stats = {}
    for cls in class_list:
        iid = cls["instructorId"]
        instructor_stats.setdefault(iid, {"name": find_professor(iid)["name"] if find_professor(iid) else f"#{iid}", "total": 0, "attended": 0.0})
        instructor_stats[iid]["total"] += 1
        key = f"{cls['day']}-{cls['slotId']}"
        if key in attendance_fractions:
            instructor_stats[iid]["attended"] += attendance_fractions[key]["fraction"]

    result = {
        "attendance_fractions": attendance_fractions,
        "totalClassesWeek": total_weekly,
        "totalClassesSemester": total_semester,
        "requiredClassesWeek": round(required_per_week, 3),
        "requiredClassesSemester": required_semester,
        "totalSelectedSemester": round(total_selected_semester, 2),
        "totalFractionalClassesWeek": round(total_frac_week, 3),
        "attendancePercentage": round(attendance_percentage, 2),
        "totalValue": round(total_value, 3),
        "avgValue": round(avg_value, 4),
        "instructorStats": instructor_stats,
        "optimal": res.success
    }
    return result

# -------------------------
# GUI
# -------------------------
class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance Optimizer - Fractional Simplex")
        self.geometry("1000x650")
        self.minsize(900, 600)

        # default profile
        self.student_profile = {"travelTime": 2.0, "timeCommitment": 0.5}
        self.timetable = {}  # key -> {day, slotId, instructorId, subject}
        self.optimized = None

        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        profile_frame = ttk.LabelFrame(top, text="Student Profile")
        profile_frame.pack(side=tk.LEFT, padx=6)

        ttk.Label(profile_frame, text="Travel Time:").grid(row=0, column=0, padx=4, pady=4)
        self.travel_cb = ttk.Combobox(profile_frame, state="readonly", width=18,
                                      values=["Under 15 min|1.0", "15-30 min|1.5", "30-60 min|2.0", "60-90 min|2.5", "Over 90 min|3.0"])
        self.travel_cb.current(2)
        self.travel_cb.grid(row=0, column=1, padx=4, pady=4)
        self.travel_cb.bind("<<ComboboxSelected>>", self._on_profile_change)

        ttk.Label(profile_frame, text="Time Commitment:").grid(row=1, column=0, padx=4, pady=4)
        self.commit_cb = ttk.Combobox(profile_frame, state="readonly", width=18,
                                      values=["No commitments|0.0", "Society/Club/Sports|0.5", "Part-time job|1.0"])
        self.commit_cb.current(1)
        self.commit_cb.grid(row=1, column=1, padx=4, pady=4)
        self.commit_cb.bind("<<ComboboxSelected>>", self._on_profile_change)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side=tk.RIGHT, padx=6)
        ttk.Button(btn_frame, text="Optimize (Fractional)", command=self._optimize_click).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Clear Timetable", command=self._clear_timetable).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="Show Instructors", command=self._show_instructors).grid(row=0, column=2, padx=4)

        # Main: left add form, right timetable
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.LabelFrame(main, text="Add Class")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text="Day:").grid(row=0, column=0, padx=6, pady=4)
        self.day_cb = ttk.Combobox(left, state="readonly", values=DAYS, width=16)
        self.day_cb.current(0)
        self.day_cb.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(left, text="Time slot:").grid(row=1, column=0, padx=6, pady=4)
        self.slot_cb = ttk.Combobox(left, state="readonly", values=[f'{s["id"]}. {s["time"]}' for s in TIME_SLOTS], width=16)
        self.slot_cb.current(0)
        self.slot_cb.grid(row=1, column=1, padx=6, pady=4)

        ttk.Label(left, text="Instructor:").grid(row=2, column=0, padx=6, pady=4)
        prof_vals = [f'{p["id"]}. {p["name"]}' for p in PROFESSORS]
        self.prof_cb = ttk.Combobox(left, state="readonly", values=prof_vals, width=30)
        self.prof_cb.current(0)
        self.prof_cb.grid(row=2, column=1, padx=6, pady=4)

        ttk.Label(left, text="Subject:").grid(row=3, column=0, padx=6, pady=4)
        self.subject_entry = ttk.Entry(left, width=28)
        self.subject_entry.grid(row=3, column=1, padx=6, pady=4)

        ttk.Button(left, text="Add Class", command=self._add_class).grid(row=4, column=0, columnspan=2, padx=6, pady=8, sticky="ew")

        # Right: timetable canvas
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.info_label = ttk.Label(right, text="Fractional attendance: 1.0 = attend all, 0.5 = attend half, etc.")
        self.info_label.pack(anchor=tk.W, padx=6, pady=(0,6))

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vscroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.table_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.table_frame, anchor='nw')
        self.table_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Headers
        ttk.Label(self.table_frame, text="Time", relief=tk.RIDGE, width=18).grid(row=0, column=0, sticky="nsew")
        for c, day in enumerate(DAYS, start=1):
            ttk.Label(self.table_frame, text=day, relief=tk.RIDGE, width=28).grid(row=0, column=c, sticky="nsew")

        self.cell_widgets = {}
        for r, slot in enumerate(TIME_SLOTS, start=1):
            ttk.Label(self.table_frame, text=slot["time"], relief=tk.RIDGE, width=18).grid(row=r, column=0, sticky="nsew")
            for c, day in enumerate(DAYS, start=1):
                cell = ttk.Frame(self.table_frame, relief=tk.GROOVE, borderwidth=1, width=220, height=42)
                cell.grid_propagate(False)
                cell.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
                lbl = ttk.Label(cell, text="(empty)", anchor=tk.CENTER)
                lbl.pack(expand=True, fill=tk.BOTH)
                key = f"{day}-{slot['id']}"
                self.cell_widgets[key] = (cell, lbl)
                # bind click to remove
                lbl.bind("<Button-1>", lambda e, k=key: self._cell_clicked(k))

        # Bottom summary
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=6)
        self.total_label = ttk.Label(bottom, text="Total Classes: 0 per week × 20 weeks = 0")
        self.total_label.pack(side=tk.LEFT, padx=6)

    def _on_profile_change(self, event=None):
        try:
            travel_val = float(self.travel_cb.get().split("|")[1])
            commit_val = float(self.commit_cb.get().split("|")[1])
            self.student_profile["travelTime"] = travel_val
            self.student_profile["timeCommitment"] = commit_val
        except Exception:
            pass

    def _add_class(self):
        day = self.day_cb.get()
        slot_text = self.slot_cb.get()
        slot_id = int(slot_text.split(".")[0])
        prof_text = self.prof_cb.get()
        prof_id = int(prof_text.split(".")[0])
        subject = self.subject_entry.get().strip() or "(no subject)"
        key = f"{day}-{slot_id}"

        if key in self.timetable:
            if not messagebox.askyesno("Overwrite?", f"A class already exists at {day} {slot_text}. Overwrite?"):
                return

        self.timetable[key] = {"day": day, "slotId": slot_id, "instructorId": prof_id, "subject": subject}
        self._update_cell_ui(key)
        self._update_total_label()
        self.subject_entry.delete(0, tk.END)

    def _cell_clicked(self, key):
        if key in self.timetable:
            if messagebox.askyesno("Remove class", f"Remove class at {key.replace('-', ' slot ')}?"):
                del self.timetable[key]
                self._update_cell_ui(key)
                self._update_total_label()

    def _update_cell_ui(self, key):
        cell, old_lbl = self.cell_widgets[key]
        for w in cell.winfo_children():
            w.destroy()

        if key in self.timetable:
            cls = self.timetable[key]
            prof = find_professor(cls["instructorId"])
            text = f"{cls['subject']}\n{prof['name'] if prof else f'Instructor #{cls['instructorId']}'}"
            label = ttk.Label(cell, text=text, anchor=tk.CENTER, justify=tk.CENTER)
            label.pack(expand=True, fill=tk.BOTH)
            label.bind("<Button-1>", lambda e, k=key: self._cell_clicked(k))
        else:
            label = ttk.Label(cell, text="(empty)", anchor=tk.CENTER)
            label.pack(expand=True, fill=tk.BOTH)
            label.bind("<Button-1>", lambda e, k=key: self._cell_clicked(k))

    def _update_total_label(self):
        total = len(self.timetable)
        self.total_label.config(text=f"Total Classes: {total} per week × 20 weeks = {total * 20}")

    def _clear_timetable(self):
        if messagebox.askyesno("Clear", "Clear entire timetable?"):
            self.timetable.clear()
            for k in list(self.cell_widgets.keys()):
                self._update_cell_ui(k)
            self._update_total_label()
            self.optimized = None
            messagebox.showinfo("Cleared", "Timetable cleared.")

    def _show_instructors(self):
        lines = [f"#{p['id']} - {p['name']}" for p in PROFESSORS]
        messagebox.showinfo("Instructors", "\n".join(lines))

    def _optimize_click(self):
        if not self.timetable:
            messagebox.showwarning("No classes", "Add classes to timetable first.")
            return

        self._on_profile_change()
        res = optimize_attendance_fractional(self.timetable, self.student_profile, weeks=20, attendance_target=0.75)
        if res is None:
            messagebox.showinfo("No solution", "Could not compute an optimal solution with current constraints.")
            return
        self.optimized = res
        self._show_optimization_result(res)

    def _show_optimization_result(self, res):
        win = tk.Toplevel(self)
        win.title("Optimization Results - Fractional Attendance")
        win.geometry("1000x650")

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(top_frame, text=f"Total Classes/Week: {res['totalClassesWeek']}", width=28).grid(row=0, column=0, padx=6)
        ttk.Label(top_frame, text=f"Total Classes/Semester: {res['totalClassesSemester']}", width=28).grid(row=0, column=1, padx=6)
        ttk.Label(top_frame, text=f"Required (75%): {res['requiredClassesWeek']}/week", width=28).grid(row=0, column=2, padx=6)
        ttk.Label(top_frame, text=f"Attending: {res['totalFractionalClassesWeek']}/week", width=28).grid(row=1, column=0, padx=6)
        ttk.Label(top_frame, text=f"Total Attending (semester): {res['totalSelectedSemester']}", width=28).grid(row=1, column=1, padx=6)
        ttk.Label(top_frame, text=f"Avg APS: {res['avgValue']}", width=28).grid(row=1, column=2, padx=6)

        # Table of weekly slots with color-coded boxes (use tk.Label for colors)
        table_frame = ttk.Frame(win)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        ttk.Label(table_frame, text="Time", relief=tk.RIDGE, width=18).grid(row=0, column=0, sticky="nsew")
        for c, day in enumerate(DAYS, start=1):
            ttk.Label(table_frame, text=day, relief=tk.RIDGE, width=28).grid(row=0, column=c, sticky="nsew")

        attendance = res["attendance_fractions"]

        for r, slot in enumerate(TIME_SLOTS, start=1):
            ttk.Label(table_frame, text=slot["time"], relief=tk.RIDGE, width=18).grid(row=r, column=0, sticky="nsew")
            for c, day in enumerate(DAYS, start=1):
                key = f"{day}-{slot['id']}"
                frame = ttk.Frame(table_frame, relief=tk.GROOVE, borderwidth=1, width=220, height=50)
                frame.grid_propagate(False)
                frame.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)

                if key in self.timetable:
                    cls = self.timetable[key]
                    # Only show Instructor number (not full name) in the final optimized grid
                    txt = f"{cls['subject']}\nInstructor {cls['instructorId']}"

                    if key in attendance:
                        fraction = attendance[key]["fraction"]
                        aps = attendance[key]["aps"]

                        # Choose background (colored boxes) and BLACK text for readability
                        if fraction >= 0.9:
                            bg_color = "#8BF28B"   # Green
                        elif fraction >= 0.7:
                            bg_color = "#F8E969"   # Yellow
                        elif fraction >= 0.4:
                            bg_color = "#F5A572"   # Orange
                        else:
                            bg_color = "#F28181"   # Red

                        fg_color = "#000000"  # Always black text

                        # Use tk.Label for colored backgrounds to work on macOS (ttk ignores bg)
                        lbl = tk.Label(frame, text=txt, anchor="center", justify="center", bg=bg_color, fg=fg_color)
                        lbl.pack(expand=True, fill="both")
                        info_lbl = tk.Label(frame, text=f"Attend: {int(fraction*100)}% (APS: {aps})",
                                            font=("TkDefaultFont", 8, "bold"), bg=bg_color, fg=fg_color)
                        info_lbl.pack()
                    else:
                        # Not selected (shouldn't usually happen if timetable contains it, but safe)
                        lbl = tk.Label(frame, text=txt + "\nNot selected (0%)", anchor="center", justify="center",
                                       bg="#ffffff", fg="#000000")
                        lbl.pack(expand=True, fill="both")
                else:
                    # empty slot
                    ttk.Label(frame, text="").pack()

        # Instructor stats treeview
        stats_frame = ttk.LabelFrame(win, text="Instructor-wise Fractional Attendance")
        stats_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)

        cols = ("weekly_frac", "semester_frac", "percent")
        tree = ttk.Treeview(stats_frame, columns=cols, show="headings", height=8)
        tree.heading("weekly_frac", text="Weekly (Fractional)")
        tree.heading("semester_frac", text="Semester (Fractional)")
        tree.heading("percent", text="Attendance %")
        tree.column("weekly_frac", width=160, anchor=tk.CENTER)
        tree.column("semester_frac", width=160, anchor=tk.CENTER)
        tree.column("percent", width=120, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)

        for iid, stats in res["instructorStats"].items():
            total = stats["total"]
            attended = stats["attended"]
            percent = round((attended / total * 100), 1) if total > 0 else 0.0
            weekly_text = f"{attended:.2f}/{total}"
            semester_text = f"{attended*20:.1f}/{total*20}"
            tree.insert("", "end", values=(weekly_text, semester_text, f"{percent}%"), text=f"#{iid} - {stats['name']}")

        # explanatory note and close
        ttk.Label(win, text="Note: Fractional attendance means attending that percentage of occurrences.\nE.g., 0.75 = attend 15 out of 20 lectures for that slot.",
                  font=("TkDefaultFont", 9, "italic")).pack(pady=6)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
