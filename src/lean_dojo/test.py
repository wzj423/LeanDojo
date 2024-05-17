from lean_dojo import *
from pathlib import Path
import json
if __name__ =="__main__":
    # repo = LeanGitRepo(url="https://github.com/rahul3613/ProofNet-lean4",commit="6eef91d99365eb6e506fa00f24d27aece609093b",local_path="/cpfs01/shared/public/public_hdd/llm_math/wuzijian/tmp/ExtractWorkspace/ProofNet-lean4/rahul3613-ProofNet-lean4-6eef91d99365eb6e506fa00f24d27aece609093b/ProofNet-lean4")
    # traced_file=TracedFile.from_traced_file("/cpfs01/shared/public/public_hdd/llm_math/wuzijian/tmp/ExtractWorkspace/ProofNet-lean4/rahul3613-ProofNet-lean4-6eef91d99365eb6e506fa00f24d27aece609093b/ProofNet-lean4",
    #                             Path("/cpfs01/shared/public/public_hdd/llm_math/wuzijian/tmp/ExtractWorkspace/ProofNet-lean4/rahul3613-ProofNet-lean4-6eef91d99365eb6e506fa00f24d27aece609093b/ProofNet-lean4/.lake/build/ir/formal/test.ast.json"),
    #                             repo)
    repo = LeanGitRepo(url="https://github.com/dwrensha/compfiles",commit="f811943507120cc0ffb7e3107189d9e0baf3f529",local_path="/cpfs01/shared/public/public_hdd/llm_math/wuzijian/miniF2f_autograde_ldj171/.cache/lean_dojo/dwrensha-compfiles-f811943507120cc0ffb7e3107189d9e0baf3f529/compfiles/")
    traced_file=TracedFile.from_traced_file("/cpfs01/shared/public/public_hdd/llm_math/wuzijian/miniF2f_autograde_ldj171/.cache/lean_dojo/dwrensha-compfiles-f811943507120cc0ffb7e3107189d9e0baf3f529/compfiles/",
                                Path("/cpfs01/shared/public/public_hdd/llm_math/wuzijian/miniF2f_autograde_ldj171/.cache/lean_dojo/dwrensha-compfiles-f811943507120cc0ffb7e3107189d9e0baf3f529/compfiles/.lake/build/ir/Compfiles/Bulgaria1998P3.ast.json"),
                                repo)
    traced_file.get_nocalc_file_text()
    traced_theorems = traced_file.get_traced_theorems()
    for tt in traced_theorems:
        sstac=tt.get_traced_smallstep_tactics()
        # clean_tac = [{"stateBefore":x["stateBefore"],"stateAfter":x["stateAfter"],"tactic":x["tactic"],"start":str(x["file_pos"]),"end":str(x["file_end_pos"]),} for x in sstac]
        clean_tac = [{
                        "proof_before":x[0],
                        "proof_after":x[1],
                        "tactic":x[2],
                        "state_before":x[3],
                        "state_after":x[4],
                        "start":str(x[5]),
                        "end":str(x[6])
                      } 
                     for x in sstac]
        # print(clean_tac)
        print(json.dumps(clean_tac,indent=4,ensure_ascii=False))