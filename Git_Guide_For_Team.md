### Basic Step-by-Step Guide for Group Members

#### 1. Clone the Entire Repository

First, group members need to copy the entire repository to their local virtual machine (VM). This will be done only once.

a. **Open Terminal (Linux/Mac) or Git Bash (Windows) or VSCode (new terminal at bash):**

b. **Navigate to the Folder You Want to Work In:**
Use the `cd` command to go to the folder where they want the project files to be. For example:

```
cd F:/course/my_project
```

c. **Clone the Repository:**
To copy the whole repository to their machine, they need to clone it using the following command:

```
git clone https://github.com/tubetraveller/heartbeat_dst.git
```

This command will create a folder called `heartbeat_dst` in their current directory, with all the project's files.

---

#### 2. Create a New Branch to Work Safely

To ensure that the work remains safe and there is no conflict with the `main` branch, group members should **create a new branch** before making any changes. This way, they can work independently without impacting the `main` branch.

a. **Navigate into the Repository Folder:**

```
cd heartbeat_dst
```

b. **Create a New Branch with Your Name:**
Each group member should create a branch named after themselves. For example, if the member's name is Asena, they could create a branch named `asena`:

```
git checkout -b asena
```

The `checkout -b` command creates and switches to a new branch.

---

#### 3. Make Changes to the Files

Group members can now make changes to the files within their new branch. They should **not work directly on the `main` branch**, as changes made there might accidentally conflict with others' work.

---

#### 4. Commit Changes Locally

After making changes, they need to save (commit) those changes locally:

If they want to add specific folders or files, they can use:

- **For a folder:**
  ```
  git add "Models/Max's model"
  ```
- **For a file:**
  ```
  git add "Models/Max's model/file_name.py"
  ```

b. **Commit Changes:**

```
git commit -m "Added updates to models"
```

Replace `"Added updates to models"` with a descriptive message about what changes were made.

---

#### 5. Push Changes to GitHub

To share their work with the rest of the team, they need to push their changes to GitHub. Since they’re working on a new branch, they need to push this new branch:

a. **Push the Branch:**

```
git push -u origin asena
```

The `-u origin` flag sets the remote repository (GitHub) as the default for future pushes.

---

#### 6. Create a Pull Request (PR)

To merge their changes back into the main branch, group members need to create a pull request (PR). This step allows others to review their changes before merging.

a. **Go to GitHub Repository Page:**
Open [GitHub repository link](https://github.com/tubetraveller/heartbeat_dst).

b. **Create a Pull Request:**

- Click on the **"Pull request"** button that should appear after pushing the branch.
- Write a realy brief summary of what changes were made.
- Submit the pull request for review.

---

#### 7. Merge Pull Request After Approval

A group reviewer will check the pull request for correctness. Once approved, it can be merged into the `main` branch.

---

#### 8. Keep Branch Updated

If multiple people are working on the project, it's essential to **update branches** to avoid conflicts:

a. **Switch Back to Main Branch:**

```
git checkout main
```

b. **Pull the Latest Changes:**

```
git pull origin main
```

c. **Switch Back to Your Feature Branch:**

```
git checkout asena
```

d. **Merge the Latest Main Changes Into Your Branch:**

```
git merge main
```

Resolve any conflicts if necessary.

---

### Best Practices

1. **Always Work in a Branch:** Avoid working directly in the `main` branch.
2. **Commit Often:** Commit changes frequently with descriptive messages.
3. **Pull Before You Push:** Always pull the latest changes before pushing to avoid conflicts.
4. **Use Feature Branches Named After Yourself:** Create branches based on your name to easily track who made which changes.
5. **Ask for Help:** If you face any issues or conflicts, it's better to ask the team for help rather than risking a broken `main` branch.

---

### Summary of Commands

1. **Clone the Repository:** only once you need to do
   ```
   git clone https://github.com/tubetraveller/heartbeat_dst.git
   ```
2. **Create a New Branch:** if you have once, contunie working over this branch 
   ```
   git checkout -b your-name
   ```
3. **Stage and Commit Changes:** after each changes (Here, each chance means that if you create/change/update a file or totaly create a folder with new files)
   ```
   git add "Models/Max's model" or git add "Models/Max's model/file_name.py"
   git commit -m "Commit message"
   ```
4. **Push the Branch to GitHub:** after each changes
   ```
   git push -u origin your-name
   ```
5. **Create a Pull Request on GitHub.** after each changes

---

This guide will help us avoid making changes directly in the `main` branch and ensure a more collaborative and error-free workflow. If you need further clarification, feel free to update the guide based on feedback from the team.

