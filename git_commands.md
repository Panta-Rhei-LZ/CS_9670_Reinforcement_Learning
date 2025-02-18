If you're working in `workspace/Lingyu` and want to merge changes into `main` or pull updates from `main`, here’s the **Git workflow** you should follow:

---

### **Working in `workspace/Lingyu` and Merging into `main`**
1. **Ensure you're on `workspace/Lingyu`**
   ```bash
   git checkout workspace/Lingyu  # or use `git switch workspace/Lingyu`
   ```
   
2. **Make Changes and Commit**
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

3. **Push Your Changes to Remote (Optional)**
   ```bash
   git push origin workspace/Lingyu
   ```

4. **Switch to `main` and Pull Latest Changes**
   ```bash
   git checkout main
   git pull origin main
   ```

5. **Merge `workspace/Lingyu` into `main`**
   ```bash
   git merge workspace/Lingyu
   ```

6. **Resolve Merge Conflicts (if any)**
   - If Git reports conflicts, manually fix them in the affected files.
   - Use `git add <file>` after fixing conflicts.
   - Then continue the merge with:
     ```bash
     git commit -m "Resolved merge conflicts"
     ```

7. **Push the Merged `main` Branch to Remote**
   ```bash
   git push origin main
   ```

---

### **Pulling Latest Updates from `main` into `workspace/Lingyu`**
If teammates make changes in `main`, you should update `workspace/Lingyu`:

1. **Ensure Your `main` is Up-to-Date**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Switch Back to `workspace/Lingyu`**
   ```bash
   git checkout workspace/Lingyu
   ```

3. **Merge `main` into `workspace/Lingyu`**
   ```bash
   git merge main
   ```

4. **Resolve Merge Conflicts (if any)** (same as before)

5. **Push the Updated `workspace/Lingyu` Branch**
   ```bash
   git push origin workspace/Lingyu
   ```

---

### **Best Practices**
- **Regularly pull from `main`** to avoid major conflicts.
- **Work in small commits** so merging is easier.
- **Use branches for features/bugs** to keep `main` stable.
- **Test before merging into `main`** (e.g., run tests or verify code).

This workflow ensures smooth collaboration and prevents conflicts. 🚀 Let me know if you need clarification!