import { componentState, fileProblem, sortFindings } from "./format";
import type { Finding } from "./types";

const f = (status: Finding["status"], code: string): Finding => ({
  code, status, class_name: "x", expected: null, confident: null, uncertain: null, message: "", evidence: {},
});

describe("sortFindings", () => {
  it("puts FAIL before REVIEW and keeps order inside a group", () => {
    const out = sortFindings([f("REVIEW", "A"), f("FAIL", "B"), f("REVIEW", "C"), f("FAIL", "D")]);
    expect(out.map((x) => x.code)).toEqual(["B", "D", "A", "C"]);
  });
  it("does not mutate its input", () => {
    const input = [f("REVIEW", "A"), f("FAIL", "B")];
    sortFindings(input);
    expect(input[0].code).toBe("A");
  });
});

describe("componentState", () => {
  it("is ok only when the confident count matches and nothing is uncertain", () => {
    expect(componentState({ expected: 2, confident: 2, uncertain: 0 })).toBe("ok");
    expect(componentState({ expected: 2, confident: 2, uncertain: 1 })).toBe("check");
    expect(componentState({ expected: 2, confident: 1, uncertain: 1 })).toBe("check");
    expect(componentState({ expected: 2, confident: 1, uncertain: 0 })).toBe("wrong");
    expect(componentState({ expected: 1, confident: 2, uncertain: 0 })).toBe("wrong");
  });
});

describe("fileProblem", () => {
  const file = (name: string, size = 10) => new File([new Uint8Array(size)], name);
  it("accepts images and rejects other types, empty and oversized files", () => {
    expect(fileProblem(file("a.PNG"))).toBeNull();
    expect(fileProblem(file("a.txt"))).toMatch(/PNG, JPG or BMP/);
    expect(fileProblem(file("a.png", 0))).toMatch(/empty/);
    expect(fileProblem(file("a.png", 101), 100)).toMatch(/25 MB/);
  });
});