def kl_div(student_logits, teacher_logits, T: float):
    # stub for KD loss
    return (student_logits - teacher_logits).abs().mean()
