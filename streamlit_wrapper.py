{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b4c930e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Press Enter to stop the Streamlit app...\n"
     ]
    }
   ],
   "source": [
    "# streamlit_wrapper.py\n",
    "import os\n",
    "from subprocess import Popen\n",
    "\n",
    "def run_app(app_file):\n",
    "    cmd = f\"streamlit run {app_file}\"\n",
    "    process = Popen(cmd, shell=True)\n",
    "    return process\n",
    "\n",
    "def stop_app(process):\n",
    "    process.kill()\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app_file = \"app.py\"  # Replace with your Streamlit app file name\n",
    "    process = run_app(app_file)\n",
    "    input(\"Press Enter to stop the Streamlit app...\")\n",
    "    stop_app(process)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc602980",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
