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

# Priority levels Option C (5-level)
PRIORITY_LEVELS = ["Very High", "High", "Medium", "Low", "Avoid"]
PRIORITY_MULTIPLIER = {
    "Very High": 1.40,
    "High": 1.25,
    "Medium": 1.10,
    "Low": 1.00,
    "Avoid": 0.80
}


# -------------------------
# Helper functions
# -------------------------
def find_professor(pid):
    return next((p for p in PROFESSORS if p["id"] == pid), None)


def calculate_aps(instructor_id, time_block, student_profile, priorities):
    """
    Base APS calculation, then scaled by priority multiplier for professor.
    """
    prof = find_professor(instructor_id)
    if prof is None:
        return 0.0
    timeBlockRatings = {"morning": 7.5, "midday": 7.0, "afternoon": 6.5}
    weights = {
        "w1": 0.1476, "w2": 0.1456, "w3": 0.1370, "w4": 0.1356,
        "w5": 0.1203, "w6": 0.1225, "w7": 0.0935, "w8": 0.0979
    }
    TB = timeBlockRatings.get(time_block, 7.0)
    HS = 5.0
    aps_base = (
        weights["w1"] * prof["pv"] +
        weights["w2"] * prof["le"] +
        weights["w3"] * prof["se"] +
        weights["w4"] * prof["ar"] +
        weights["w5"] * TB +
        weights["w6"] * HS -
        weights["w7"] * student_profile.get("travelTime", 0.0) -
        weights["w8"] * student_profile.get("timeCommitment", 0.0)
    )
    # Apply priority multiplier (default Medium if not found)
    level = priorities.get(instructor_id, "Medium")
    mult = PRIORITY_MULTIPLIER.get(level, 1.10)
    aps_final = aps_base * mult
    return round(aps_final, 3)


def optimize_attendance_simplex(timetable, student_profile, priorities, desired_attendance_percent):
    """
    Linear programming:
    - Fractional attendance per class (variables between 0 and 1)
    - Sum of fractions per week == desired_percent * total_classes_per_week
    - Objective: maximize sum(APS * fraction) -> linprog minimizes so we negate APS
    - Optional instructor-based minimum constraint (kept similar to earlier: if instructor has >=2 classes, try to keep >=2 occurrences)
    """
    total_classes_per_week = len(timetable)
    if total_classes_per_week == 0:
        return None

    # Normalize desired attendance percent to [0,1]
    d_pct = max(0.0, min(100.0, float(desired_attendance_percent))) / 100.0

    # Required per week (can be fractional, since fractional attendance is allowed)
    required_per_week = total_classes_per_week * d_pct

    # Semester totals (20 weeks assumed)
    weeks = 20
    total_classes_semester = total_classes_per_week * weeks
    required_classes_semester = int(math.ceil(required_per_week * weeks))

    # Build class list and APS values
    class_list = []
    for key, cls in timetable.items():
        slot = next((s for s in TIME_SLOTS if s["id"] == cls["slotId"]), None)
        aps = calculate_aps(cls["instructorId"], slot["block"] if slot else "midday", student_profile, priorities)
        class_list.append({**cls, "aps": aps, "key": key})
    n = len(class_list)
    if n == 0:
        return None

    # Objective coefficients (minimize) -> use negative APS to maximize
    c = [-cls["aps"] for cls in class_list]

    # Equality constraint: sum(x_i) = required_per_week
    A_eq = [[1.0] * n]
    b_eq = [required_per_week]

    # Instructor minimum constraint: if instructor has >=2 classes in week, try to ensure at least 2 aggregated (as in original).
    # Implement as A_ub with -sum >= -2 -> -(sum) <= -2 (same structure)
    A_ub = []
    b_ub = []
    instructor_classes = {}
    for i, cls in enumerate(class_list):
        iid = cls["instructorId"]
        instructor_classes.setdefault(iid, []).append(i)
    for iid, indices in instructor_classes.items():
        if len(indices) >= 2:
            row = [0.0] * n
            for idx in indices:
                row[idx] = -1.0
            A_ub.append(row)
            b_ub.append(-2.0)

    # Variable bounds 0..1
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Solve
    result = linprog(c=c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None,
                     A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # If solver fails with instructor constraints, relax them
    if not result.success and A_ub:
        result = linprog(c=c, A_ub=None, b_ub=None, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if not result.success:
            return None

    # Build attendance fractions
    attendance_fractions = {}
    for i, val in enumerate(result.x):
        cls = class_list[i]
        key = f"{cls['day']}-{cls['slotId']}"
        attendance_fractions[key] = {
            **cls,
            "fraction": round(float(val), 3),
            "aps_weighted": round(float(val) * cls["aps"], 3),
            "aps": cls["aps"]
        }

    total_fractional_classes = sum(af["fraction"] for af in attendance_fractions.values())
    total_selected_semester = total_fractional_classes * weeks
    attendance_percentage = (total_selected_semester / total_classes_semester * 100.0) if total_classes_semester > 0 else 0.0
    total_value = sum(af["aps_weighted"] for af in attendance_fractions.values())
    avg_value = (total_value / total_fractional_classes) if total_fractional_classes > 0 else 0.0

    # Per-instructor stats
    instructor_stats = {}
    for cls in class_list:
        iid = cls["instructorId"]
        if iid not in instructor_stats:
            # store name in stats internally, but it will not be shown in final output
            prof = find_professor(iid)
            instructor_stats[iid] = {"name": prof["name"] if prof else f"#{iid}", "total": 0, "attended": 0.0}
        instructor_stats[iid]["total"] += 1
        key = f"{cls['day']}-{cls['slotId']}"
        if key in attendance_fractions:
            instructor_stats[iid]["attended"] += attendance_fractions[key]["fraction"]

    return {
        "attendance_fractions": attendance_fractions,
        "totalClassesWeek": total_classes_per_week,
        "totalClassesSemester": total_classes_semester,
        "requiredClassesSemester": required_classes_semester,
        "requiredClassesWeek": round(required_per_week, 3),
        "totalSelectedSemester": round(total_selected_semester, 2),
        "totalFractionalClassesWeek": round(total_fractional_classes, 3),
        "attendancePercentage": round(attendance_percentage, 2),
        "totalValue": round(total_value, 3),
        "avgValue": round(avg_value, 3),
        "instructorStats": instructor_stats,
        "optimal": result.success
    }


# -------------------------
# Tkinter GUI
# -------------------------
class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance Optimizer - Fractional Simplex (With Priorities)")
        self.geometry("1100x700")
        self.resizable(True, True)

        # default student profile
        self.student_profile = {"travelTime": 2.0, "timeCommitment": 0.5}

        # timetable: keys like "Monday-1" -> { day, slotId, instructorId, subject }
        self.timetable = {}

        # priorities: instructor_id -> priority_level_string
        # default to Medium for all
        self.priorities = {p["id"]: "Medium" for p in PROFESSORS}

        # desired attendance percent (default 75)
        self.desired_attendance_percent = tk.DoubleVar(value=75.0)

        self.optimized = None
        self._build_ui()

    def _build_ui(self):
        # Top frame
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        # Student profile box
        profile_frame = ttk.LabelFrame(top, text="Student Profile / Settings")
        profile_frame.pack(side=tk.LEFT, padx=6, pady=2, fill=tk.Y)

        ttk.Label(profile_frame, text="Travel Time:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.travel_cb = ttk.Combobox(profile_frame, values=[
            "Under 15 min|1.0", "15-30 min|1.5", "30-60 min|2.0",
            "60-90 min|2.5", "Over 90 min|3.0"
        ], state="readonly", width=20)
        self.travel_cb.current(2)
        self.travel_cb.grid(row=0, column=1, padx=4, pady=4)
        self.travel_cb.bind("<<ComboboxSelected>>", self._on_profile_change)

        ttk.Label(profile_frame, text="Time Commitment:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.commit_cb = ttk.Combobox(profile_frame, values=[
            "No commitments|0.0", "Society/Club/Sports|0.5", "Part-time job|1.0"
        ], state="readonly", width=20)
        self.commit_cb.current(1)
        self.commit_cb.grid(row=1, column=1, padx=4, pady=4)
        self.commit_cb.bind("<<ComboboxSelected>>", self._on_profile_change)

        # Desired attendance percent
        ttk.Label(profile_frame, text="Desired Attendance (%):").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        self.desired_entry = ttk.Entry(profile_frame, width=8, textvariable=self.desired_attendance_percent)
        self.desired_entry.grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)

        # Set priorities button
        ttk.Button(profile_frame, text="Set Professor Priorities", command=self._open_priorities_window).grid(row=3, column=0, columnspan=2, pady=6, padx=4, sticky=tk.EW)

        # Action buttons
        btn_frame = ttk.Frame(top)
        btn_frame.pack(side=tk.RIGHT, padx=6)
        ttk.Button(btn_frame, text="Optimize (Fractional)", command=self._optimize_click).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(btn_frame, text="Clear Timetable", command=self._clear_timetable).grid(row=0, column=1, padx=6, pady=4)
        ttk.Button(btn_frame, text="Show Instructors", command=self._show_instructors).grid(row=0, column=2, padx=6, pady=4)

        # Main area
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Left: Add class form
        left = ttk.LabelFrame(main, text="Add Class")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text="Day:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        self.day_cb = ttk.Combobox(left, values=DAYS, state="readonly", width=18)
        self.day_cb.current(0)
        self.day_cb.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(left, text="Time slot:").grid(row=1, column=0, sticky=tk.W, padx=6, pady=4)
        self.slot_cb = ttk.Combobox(left, values=[f'{s["id"]}. {s["time"]}' for s in TIME_SLOTS], state="readonly", width=18)
        self.slot_cb.current(0)
        self.slot_cb.grid(row=1, column=1, padx=6, pady=4)

        ttk.Label(left, text="Instructor:").grid(row=2, column=0, sticky=tk.W, padx=6, pady=4)
        prof_vals = [f'{p["id"]}. {p["name"]}' for p in PROFESSORS]
        self.prof_cb = ttk.Combobox(left, values=prof_vals, state="readonly", width=28)
        self.prof_cb.current(0)
        self.prof_cb.grid(row=2, column=1, padx=6, pady=4, columnspan=2)

        ttk.Label(left, text="Subject:").grid(row=3, column=0, sticky=tk.W, padx=6, pady=4)
        self.subject_entry = ttk.Entry(left, width=28)
        self.subject_entry.grid(row=3, column=1, padx=6, pady=4, columnspan=2)

        ttk.Button(left, text="Add Class", command=self._add_class).grid(row=4, column=0, columnspan=3, padx=6, pady=8, sticky=tk.EW)

        # Right: Timetable grid
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.info_label = ttk.Label(right, text="Fractional attendance: 1.0 = attend all, 0.5 = attend half, etc.")
        self.info_label.pack(anchor=tk.W, padx=6, pady=(0, 6))

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

        # Build headers and empty cells
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
                lbl.bind("<Button-1>", lambda e, k=key: self._cell_clicked(k))

        # Bottom metrics
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=10, pady=6)
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
        cell, _ = self.cell_widgets[key]
        for widget in cell.winfo_children():
            widget.destroy()
        if key in self.timetable:
            cls = self.timetable[key]
            prof = find_professor(cls["instructorId"])
            # Keep professor names visible in the main timetable (editable UI)
            text = f"{cls['subject']}\n{prof['name'] if prof else f'Instructor #{cls['instructorId']}'}"
            lbl2 = ttk.Label(cell, text=text, anchor=tk.CENTER, justify=tk.CENTER)
            lbl2.pack(expand=True, fill=tk.BOTH)
            lbl2.bind("<Button-1>", lambda e, k=key: self._cell_clicked(k))
            lbl2.configure(background="#e8f4ff")
            cell.configure(width=220, height=42)
        else:
            lbl2 = ttk.Label(cell, text="(empty)", anchor=tk.CENTER)
            lbl2.pack(expand=True, fill=tk.BOTH)
            lbl2.bind("<Button-1>", lambda e, k=key: self._cell_clicked(k))
            lbl2.configure(background="#ffffff")
            cell.configure(width=220, height=42)

    def _update_total_label(self):
        total = len(self.timetable)
        self.total_label.config(text=f"Total Classes: {total} per week × 20 weeks = {total * 20}")

    def _clear_timetable(self):
        if messagebox.askyesno("Clear", "Clear the entire timetable?"):
            self.timetable.clear()
            for k in list(self.cell_widgets.keys()):
                self._update_cell_ui(k)
            self._update_total_label()
            self.optimized = None
            messagebox.showinfo("Cleared", "Timetable cleared.")

    def _show_instructors(self):
        lines = []
        for p in PROFESSORS:
            level = self.priorities.get(p["id"], "Medium")
            lines.append(f"#{p['id']} - {p['name']}  (Priority: {level})")
        messagebox.showinfo("Instructors", "\n".join(lines))

    def _open_priorities_window(self):
        # Option 1 layout (compact scrolling list)
        win = tk.Toplevel(self)
        win.title("Set Professor Priorities")
        win.geometry("420x520")

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        canvas = tk.Canvas(frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=vscroll.set)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Keep references to combobox widgets so we can read them on save
        combobox_refs = {}

        for i, prof in enumerate(PROFESSORS):
            row = ttk.Frame(inner)
            row.pack(fill=tk.X, pady=4, padx=4)
            ttk.Label(row, text=f"{prof['name']}", width=30).pack(side=tk.LEFT, padx=(0, 6))
            cb = ttk.Combobox(row, values=PRIORITY_LEVELS, state="readonly", width=12)
            cb.set(self.priorities.get(prof["id"], "Medium"))
            cb.pack(side=tk.LEFT)
            combobox_refs[prof["id"]] = cb

        def _save_priorities():
            for pid, cb in combobox_refs.items():
                val = cb.get()
                if val not in PRIORITY_MULTIPLIER:
                    val = "Medium"
                self.priorities[pid] = val
            win.destroy()
            messagebox.showinfo("Saved", "Professor priorities updated.")

        ttk.Button(win, text="Save Priorities", command=_save_priorities).pack(pady=8)

    def _optimize_click(self):
        if not self.timetable:
            messagebox.showwarning("No classes", "Add some classes first.")
            return
        self._on_profile_change()
        # read desired attendance percent from entry
        try:
            dval = float(self.desired_attendance_percent.get())
            if dval < 0 or dval > 100:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid input", "Enter desired attendance percent between 0 and 100.")
            return

        result = optimize_attendance_simplex(self.timetable, self.student_profile, self.priorities, dval)
        if result is None:
            messagebox.showinfo("No solution", "Could not find optimal solution for the given constraints.")
            return
        self.optimized = result
        self._show_optimization_result(result)

    def _show_optimization_result(self, res):
        win = tk.Toplevel(self)
        win.title("Optimization Results - Fractional Attendance")
        win.geometry("1000x700")

        top_frame = ttk.Frame(win)
        top_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top_frame, text=f"Total Classes/Week: {res['totalClassesWeek']}", width=30).grid(row=0, column=0, padx=6)
        ttk.Label(top_frame, text=f"Total Classes/Semester: {res['totalClassesSemester']}", width=30).grid(row=0, column=1, padx=6)
        ttk.Label(top_frame, text=f"Required ({self.desired_attendance_percent.get()}%): {res['requiredClassesWeek']}/week", width=30).grid(row=0, column=2, padx=6)
        ttk.Label(top_frame, text=f"Attending (week fractional): {res['totalFractionalClassesWeek']}/week", width=30).grid(row=1, column=0, padx=6)
        ttk.Label(top_frame, text=f"Total Attending (semester): {res['totalSelectedSemester']}", width=30).grid(row=1, column=1, padx=6)
        ttk.Label(top_frame, text=f"Attendance % (semester): {res['attendancePercentage']}%", width=30).grid(row=1, column=2, padx=6)
        ttk.Label(top_frame, text=f"Avg APS: {res['avgValue']}", width=30).grid(row=2, column=1, padx=6)

        # Weekly pattern with fractions
        table_frame = ttk.Frame(win)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        ttk.Label(table_frame, text="Time", relief=tk.RIDGE, width=16).grid(row=0, column=0, sticky="nsew")
        for c, day in enumerate(DAYS, start=1):
            ttk.Label(table_frame, text=day, relief=tk.RIDGE, width=28).grid(row=0, column=c, sticky="nsew")

        attendance_fractions = res["attendance_fractions"]
        for r, slot in enumerate(TIME_SLOTS, start=1):
            ttk.Label(table_frame, text=slot["time"], relief=tk.RIDGE, width=16).grid(row=r, column=0, sticky="nsew")
            for c, day in enumerate(DAYS, start=1):
                key = f"{day}-{slot['id']}"
                frame = ttk.Frame(table_frame, relief=tk.GROOVE, borderwidth=1, width=220, height=60)
                frame.grid_propagate(False)
                frame.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)

                if key in self.timetable:
                    cls = self.timetable[key]
                    # FINAL OUTPUT: show Instructor number only, no professor name
                    txt = f"{cls['subject']}\nInstructor {cls['instructorId']}"
                    if key in attendance_fractions:
                        fraction = attendance_fractions[key]["fraction"]
                        aps = attendance_fractions[key]["aps"]
                        # simple color gradient logic
                        if fraction >= 0.9:
                            bg_color = "#d1fae5"
                            fg_color = "#065f46"
                        elif fraction >= 0.7:
                            bg_color = "#fef3c7"
                            fg_color = "#78350f"
                        elif fraction >= 0.4:
                            bg_color = "#fed7aa"
                            fg_color = "#7c2d12"
                        else:
                            bg_color = "#fecaca"
                            fg_color = "#7f1d1d"
                        lbl = tk.Label(frame, text=txt, anchor=tk.CENTER, justify=tk.CENTER,
                                       bg=bg_color, fg=fg_color)
                        lbl.pack(expand=True, fill=tk.BOTH)
                        info_lbl = tk.Label(frame, text=f"Attend: {int(fraction*100)}% (APS: {aps})",
                                            font=("TkDefaultFont", 8, "bold"),
                                            bg=bg_color, fg=fg_color)
                        info_lbl.pack()
                    else:
                        lbl = tk.Label(frame, text=txt, anchor=tk.CENTER, justify=tk.CENTER, bg="#f3f4f6", fg="#6b7280")
                        lbl.pack(expand=True, fill=tk.BOTH)
                        tk.Label(frame, text="Not selected (0%)", font=("TkDefaultFont", 8), bg="#f3f4f6", fg="#6b7280").pack()
                else:
                    ttk.Label(frame, text="").pack()

        # Instructor stats
        stats_frame = ttk.LabelFrame(win, text="Instructor-wise Fractional Attendance")
        stats_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)
        tree = ttk.Treeview(stats_frame, columns=("weekly_frac", "semester_frac", "percent"), show="tree headings", height=8)
        tree.heading("#0", text="Instructor")
        tree.heading("weekly_frac", text="Weekly (Fractional)")
        tree.heading("semester_frac", text="Semester (Fractional)")
        tree.heading("percent", text="Attendance %")
        tree.column("#0", width=280, anchor=tk.W)
        tree.column("weekly_frac", width=150, anchor=tk.CENTER)
        tree.column("semester_frac", width=150, anchor=tk.CENTER)
        tree.column("percent", width=120, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)

        # FINAL OUTPUT: use "Instructor <id>" as the label (no professor names)
        for iid, stats in res["instructorStats"].items():
            total = stats["total"]
            attended = stats["attended"]
            percent = round((attended / total * 100), 1) if total > 0 else 0.0
            tree.insert("", "end", text=f"Instructor {iid}",
                        values=(f"{attended:.2f}/{total}",
                                f"{attended*20:.1f}/{total*20}",
                                f"{percent}%"))

        ttk.Label(win, text="Note: Fractional attendance means attending that fraction of occurrences.\n"
                             "E.g., 0.75 = attend 15 out of 20 lectures for that slot.",
                  font=("TkDefaultFont", 9, "italic")).pack(pady=4)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)


if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
