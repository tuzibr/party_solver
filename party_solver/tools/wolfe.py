import math

def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    zero = 0.0
    p66 = 0.66
    two = 2.0
    three = 3.0

    # 检查 dx 是否为 0
    if dx == 0.0:
        sgnd = 0.0
    else:
        sgnd = dp * (dx / abs(dx))

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.
    if fp > fx:
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / two) * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / two
        brackt = True

    # Second case: A lower function value and derivatives of opposite
    # sign. The minimum is bracketed. If the cubic step is farther from
    # stp than the secant step, the cubic step is taken, otherwise the
    # secant step is taken.
    elif sgnd < zero:
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True

    # Third case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative decreases.
    elif abs(dp) < abs(dx):
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * math.sqrt(max(zero, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < zero and gamma != zero:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + p66 * (sty - stp), stpf)
            else:
                stpf = max(stp + p66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)

    # Fourth case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative does not decrease. If the
    # minimum is not bracketed, the step is either stpmin or stpmax,
    # otherwise the cubic step is taken.
    else:
        if brackt:
            theta = three * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * math.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < zero:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new step.
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, brackt

def dcsrch(stp, f, g, ftol, gtol, xtol, task, stpmin, stpmax, isave, dsave):
    zero = 0.0
    p5 = 0.5
    p66 = 0.66
    xtrapl = 1.1
    xtrapu = 4.0


    if task[:5] == b'START':
        # Check the input arguments for errors
        if stp < stpmin:
            task = b'ERROR: STP .LT. STPMIN'
        if stp > stpmax:
            task = b'ERROR: STP .GT. STPMAX'
        if g >= zero:
            task = b'ERROR: INITIAL G .GE. ZERO'
        if ftol < zero:
            task = b'ERROR: FTOL .LT. ZERO'
        if gtol < zero:
            task = b'ERROR: GTOL .LT. ZERO'
        if xtol < zero:
            task = b'ERROR: XTOL .LT. ZERO'
        if stpmin < zero:
            task = b'ERROR: STPMIN .LT. ZERO'
        if stpmax < stpmin:
            task = b'ERROR: STPMAX .LT. STPMIN'

        # Exit if there are errors on input
        if task[:5] == b'ERROR':
            return stp, f, g, task

        brackt = False
        stage = 1
        finit = f
        ginit = g
        gtest = ftol * ginit
        width = stpmax - stpmin
        width1 = width / p5

        stx = zero
        fx = finit
        gx = ginit
        sty = zero
        fy = finit
        gy = ginit
        stmin = zero
        stmax = stp + xtrapu * stp

        task = b'FG'

        # Save local variables
        isave[0] = 1 if brackt else 0
        isave[1] = stage
        dsave[0] = ginit
        dsave[1] = gtest
        dsave[2] = gx
        dsave[3] = gy
        dsave[4] = finit
        dsave[5] = fx
        dsave[6] = fy
        dsave[7] = stx
        dsave[8] = sty
        dsave[9] = stmin
        dsave[10] = stmax
        dsave[11] = width
        dsave[12] = width1

        return stp, f, g, task
    else:
        # Restore local variables
        brackt = isave[0] == 1
        stage = isave[1]
        ginit = dsave[0]
        gtest = dsave[1]
        gx = dsave[2]
        gy = dsave[3]
        finit = dsave[4]
        fx = dsave[5]
        fy = dsave[6]
        stx = dsave[7]
        sty = dsave[8]
        stmin = dsave[9]
        stmax = dsave[10]
        width = dsave[11]
        width1 = dsave[12]

    # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the algorithm enters the second stage
    ftest = finit + stp * gtest
    if stage == 1 and f <= ftest and g >= zero:
        stage = 2

    # Test for warnings
    if brackt and (stp <= stmin or stp >= stmax):
        task = b'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
    if brackt and stmax - stmin <= xtol * stmax:
        task = b'WARNING: XTOL TEST SATISFIED'
    if stp == stpmax and f <= ftest and g <= gtest:
        task = b'WARNING: STP = STPMAX'
    if stp == stpmin and (f > ftest or g >= gtest):
        task = b'WARNING: STP = STPMIN'

    # Test for convergence
    if f <= ftest and abs(g) <= gtol * (-ginit):
        task = b'CONVERGENCE'

    # Test for termination
    if task[:4] == b'WARN' or task[:4] == b'CONV':
        pass
    else:
        # A modified function is used to predict the step during the first stage if a lower function value has been obtained but the decrease is not sufficient
        if stage == 1 and f <= fx and f > ftest:
            fm = f - stp * gtest
            fxm = fx - stx * gtest
            fym = fy - sty * gtest
            gm = g - gtest
            gxm = gx - gtest
            gym = gy - gtest
            # print(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)
            stx, fxm, gxm, sty, fym, gym, stp, brackt = dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)
            # print(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)
            fx = fxm + stx * gtest
            fy = fym + sty * gtest
            gx = gxm + gtest
            gy = gym + gtest
        else:
            stx, fx, gx, sty, fy, gy, stp, brackt = dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)

        # Decide if a bisection step is needed
        if brackt:
            if abs(sty - stx) >= p66 * width1:
                stp = stx + p5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)

        # Set the minimum and maximum steps allowed for stp
        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stp + xtrapl * (stp - stx)
            stmax = stp + xtrapu * (stp - stx)

        # Force the step to be within the bounds stpmax and stpmin
        stp = max(stp, stpmin)
        stp = min(stp, stpmax)

        # If further progress is not possible, let stp be the best point obtained during the search
        if brackt and (stp <= stmin or stp >= stmax) or (brackt and stmax - stmin <= xtol * stmax):
            stp = stx

        # Obtain another function and derivative
        task = b'FG'

    # Save local variables
    isave[0] = 1 if brackt else 0
    isave[1] = stage
    dsave[0] = ginit
    dsave[1] = gtest
    dsave[2] = gx
    dsave[3] = gy
    dsave[4] = finit
    dsave[5] = fx
    dsave[6] = fy
    dsave[7] = stx
    dsave[8] = sty
    dsave[9] = stmin
    dsave[10] = stmax
    dsave[11] = width
    dsave[12] = width1

    return stp, f, g, task